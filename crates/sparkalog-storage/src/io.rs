use crate::{Column, Relation};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum LoadError {
    GitLfsPointer(PathBuf),
    InvalidFieldCount {
        line: usize,
    },
    InvalidU32 {
        line: usize,
        column: usize,
        value: String,
    },
    Io(io::Error),
    Storage(crate::Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GitLfsPointer(path) => write!(
                formatter,
                "{} is a Git LFS pointer, not materialized graph data",
                path.display()
            ),
            Self::InvalidFieldCount { line } => {
                write!(formatter, "line {line} does not contain exactly two fields")
            }
            Self::InvalidU32 {
                line,
                column,
                value,
            } => write!(
                formatter,
                "line {line}, column {column} is not a u32: {value:?}"
            ),
            Self::Io(error) => error.fmt(formatter),
            Self::Storage(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Storage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for LoadError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<crate::Error> for LoadError {
    fn from(error: crate::Error) -> Self {
        Self::Storage(error)
    }
}

/// Load a whitespace-separated binary `u32` relation directly into canonical
/// CUDA-managed columns. The file is read twice so ingestion does not require
/// a duplicate heap-resident copy of the relation.
pub fn load_binary_u32(path: impl AsRef<Path>) -> Result<Relation, LoadError> {
    let path = path.as_ref();
    let rows = count_rows(BufReader::new(File::open(path)?), path)?;
    let mut left = Column::new_filled(rows, 0)?;
    let mut right = Column::new_filled(rows, 0)?;
    populate_columns(
        BufReader::new(File::open(path)?),
        left.as_mut_slice(),
        right.as_mut_slice(),
    )?;
    Ok(Relation::from_columns(vec![left, right])?)
}

fn data_line(line: &str) -> Option<&str> {
    let line = line.trim();
    (!line.is_empty() && !line.starts_with('#') && !line.starts_with('%')).then_some(line)
}

fn count_rows(reader: impl BufRead, path: &Path) -> Result<usize, LoadError> {
    let mut rows = 0;
    for line in reader.lines() {
        let line = line?;
        let Some(line) = data_line(&line) else {
            continue;
        };
        if line == "version https://git-lfs.github.com/spec/v1" {
            return Err(LoadError::GitLfsPointer(path.to_owned()));
        }
        rows += 1;
    }
    Ok(rows)
}

fn populate_columns(
    reader: impl BufRead,
    left: &mut [u32],
    right: &mut [u32],
) -> Result<(), LoadError> {
    let mut row = 0;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let Some(line) = data_line(&line) else {
            continue;
        };
        let mut fields = line.split_whitespace();
        let first = fields.next();
        let second = fields.next();
        if first.is_none() || second.is_none() || fields.next().is_some() {
            return Err(LoadError::InvalidFieldCount {
                line: line_index + 1,
            });
        }
        left[row] = parse_u32(first.unwrap(), line_index + 1, 1)?;
        right[row] = parse_u32(second.unwrap(), line_index + 1, 2)?;
        row += 1;
    }
    debug_assert_eq!(row, left.len());
    Ok(())
}

fn parse_u32(value: &str, line: usize, column: usize) -> Result<u32, LoadError> {
    value.parse().map_err(|_| LoadError::InvalidU32 {
        line,
        column,
        value: value.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn counts_and_populates_binary_facts() {
        let input = b"# source destination\n1\t2\n\n3 4\n% comment\n5\t6\n";
        let rows = count_rows(Cursor::new(input), Path::new("memory")).unwrap();
        let mut left = vec![0; rows];
        let mut right = vec![0; rows];
        populate_columns(Cursor::new(input), &mut left, &mut right).unwrap();

        assert_eq!(left, [1, 3, 5]);
        assert_eq!(right, [2, 4, 6]);
    }

    #[test]
    fn recognizes_unmaterialized_lfs_data() {
        let input = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n";
        assert!(matches!(
            count_rows(Cursor::new(input), Path::new("edge.facts")),
            Err(LoadError::GitLfsPointer(_))
        ));
    }

    #[test]
    fn rejects_non_binary_rows() {
        let mut left = [0];
        let mut right = [0];
        assert!(matches!(
            populate_columns(Cursor::new(b"1 2 3\n"), &mut left, &mut right),
            Err(LoadError::InvalidFieldCount { line: 1 })
        ));
    }
}
