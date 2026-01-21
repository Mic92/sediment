//! Content type for chunking
//!
//! This module is kept for the ContentType enum used by the chunker.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::error::{AlectoError, Result};

/// Content type of content (used for smart chunking)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ContentType {
    /// Plain text content
    #[default]
    Text,
    /// Markdown formatted content
    Markdown,
    /// JSON data
    Json,
    /// YAML data
    Yaml,
    /// Source code
    Code,
}

impl ContentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ContentType::Text => "text",
            ContentType::Markdown => "markdown",
            ContentType::Json => "json",
            ContentType::Yaml => "yaml",
            ContentType::Code => "code",
        }
    }
}

impl FromStr for ContentType {
    type Err = AlectoError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "text" => Ok(ContentType::Text),
            "markdown" | "md" => Ok(ContentType::Markdown),
            "json" => Ok(ContentType::Json),
            "yaml" | "yml" => Ok(ContentType::Yaml),
            "code" => Ok(ContentType::Code),
            _ => Err(AlectoError::InvalidContentType(s.to_string())),
        }
    }
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
