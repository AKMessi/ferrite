#![allow(dead_code)]
use byteorder::{LittleEndian, ReadBytesExt};
use std::{
    collections::HashMap, fs::File, io::{BufReader, Read, Seek}, path::Path
};
use thiserror::Error;

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid utf-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("invalid magic bytes — not a GGUF file")]
    InvalidMagic,

    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("unknown metadata type tag: {0}")]
    UnknownMetadataType(u32),

    #[error("invalid bool byte: {0}")]
    InvalidBool(u8),

    #[error("{what} length {len} is too large for this machine")]
    TooLarge { what: &'static str, len: u64 },

    #[error("tensor not found")]
    TensorNotFound,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::U32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::Str(v) => Some(v.as_str()),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::U64(v) => Some(*v),
            MetadataValue::U32(v) => Some(*v as u64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    pub data_offset: u64,
    pub file_offset: u64,
}

pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "UNKNOWN",
    }
}

#[derive(Debug)]
pub struct GGUFFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,

    // Absolute file offset where tensor data begins.
    pub tensor_data_start: u64,
    pub alignment: u64,
}

impl GGUFFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GGUFError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        parse(&mut reader)
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<GGUFFile, GGUFError> {
    // [4] magic u32  — must be 0x46554747
    // [4] version u32
    // [8] tensor_count u64
    // [8] metadata_kv_count u64

    // parsing the first four bytes of the gguf file to check whether the file is valid, as they are always the letters G, G, U, F in little-endian format
    let magic = reader.read_u32::<LittleEndian>()?;

    if magic != GGUF_MAGIC {
        return Err(GGUFError::InvalidMagic);
    }

    // parsing the version of the gguf file
    let version = reader.read_u32::<LittleEndian>()?;
    if version < 1 || version > 3 {
        return Err(GGUFError::UnsupportedVersion(version));
    }

    // checking the tensor and metadata count to tell the parser how many things to expect
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let metadata_kv_count = reader.read_u64::<LittleEndian>()?;

    let mut metadata = HashMap::new();

    for _ in 0..metadata_kv_count {
        let key = read_string(reader)?;
        let value = read_metadata_value(reader)?;
        metadata.insert(key, value);
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as u64;

    let mut tensor_infos_raw = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(reader)?;
        let n_dims = reader.read_u32::<LittleEndian>()?;

        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(reader.read_u64::<LittleEndian>()?);
        }

        shape.reverse();

        let ggml_type = reader.read_u32::<LittleEndian>()?;
        let data_offset = reader.read_u64::<LittleEndian>()?;

        tensor_infos_raw.push((name, shape, ggml_type, data_offset));
    }

    let current_pos = reader.stream_position()?;
    let tensor_data_start = align_offset(current_pos, alignment);

    let tensors = tensor_infos_raw
        .into_iter()
        .map(|(name, shape, ggml_type, data_offset)| TensorInfo {
            file_offset: tensor_data_start + data_offset,
            name,
            shape,
            ggml_type,
            data_offset,
        })
        .collect();

    Ok(GGUFFile {
        version,
        tensor_count,
        metadata_kv_count,
        metadata,
        tensors,
        tensor_data_start,
        alignment,
    })
}

// helper functions

fn read_string<R: Read>(reader: &mut R) -> Result<String, GGUFError> {
    let len = reader.read_u64::<LittleEndian>()?;

    // guard against corrupt files
    if len > 1_000_000 {
        return Err(GGUFError::TooLarge {
            what: "string",
            len: len,
        });
    }

    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_metadata_value<R: Read + Seek>(reader: &mut R) -> Result<MetadataValue, GGUFError> {
    let type_tag = reader.read_u32::<LittleEndian>()?;
    read_value_of_type(reader, type_tag)
}

fn read_value_of_type<R: Read + Seek>(
    reader: &mut R,
    type_tag: u32,
) -> Result<MetadataValue, GGUFError> {
    match type_tag {
        0 => Ok(MetadataValue::U8(reader.read_u8()?)),
        1 => Ok(MetadataValue::I8(reader.read_i8()?)),
        2 => Ok(MetadataValue::U16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::I16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::U32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::I32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::F32(reader.read_f32::<LittleEndian>()?)),
        7 => {
            let b = reader.read_u8()?;
            match b {
                0 => Ok(MetadataValue::Bool(false)),
                1 => Ok(MetadataValue::Bool(true)),
                _ => Err(GGUFError::InvalidBool(b)),
            }
        }
        8 => Ok(MetadataValue::Str(read_string(reader)?)),
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;

            if count > 1_000_000 {
                return Err(GGUFError::TooLarge {
                    what: "array",
                    len: count,
                });
            }

            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_value_of_type(reader, elem_type)?);
            }
            Ok(MetadataValue::Array(items))
        }
        10 => Ok(MetadataValue::U64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::I64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::F64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(GGUFError::UnknownMetadataType(type_tag)),
    }
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) / alignment * alignment
}
