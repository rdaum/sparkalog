use rayon::prelude::*;

use crate::InternedValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelimitedOptions {
    pub delimiter: u8,
    pub has_header: bool,
}

impl Default for DelimitedOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: false,
        }
    }
}

pub fn parse_delimited_parallel(
    source: &str,
    options: DelimitedOptions,
) -> Result<Vec<Vec<InternedValue>>, DelimitedError> {
    if options.delimiter.is_ascii_whitespace() && options.delimiter != b'\t' {
        return Err(DelimitedError {
            line: 0,
            column: 0,
            message: "delimiter must be a visible ASCII byte or tab".into(),
        });
    }
    let skip = usize::from(options.has_header);
    source
        .lines()
        .collect::<Vec<_>>()
        .into_par_iter()
        .enumerate()
        .skip(skip)
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line, record)| parse_record(record, line + 1, options.delimiter))
        .collect()
}

fn parse_record(
    record: &str,
    line: usize,
    delimiter: u8,
) -> Result<Vec<InternedValue>, DelimitedError> {
    let bytes = record.as_bytes();
    let mut fields = Vec::new();
    let mut position = 0;
    loop {
        let column = position + 1;
        if bytes.get(position) == Some(&b'"') {
            position += 1;
            let mut value = String::new();
            let mut closed = false;
            while position < bytes.len() {
                if bytes[position] == b'"' {
                    if bytes.get(position + 1) == Some(&b'"') {
                        value.push('"');
                        position += 2;
                    } else {
                        position += 1;
                        closed = true;
                        break;
                    }
                } else {
                    let character = record[position..]
                        .chars()
                        .next()
                        .expect("record is not exhausted");
                    value.push(character);
                    position += character.len_utf8();
                }
            }
            if !closed {
                return Err(DelimitedError {
                    line,
                    column,
                    message: "unterminated quoted field".into(),
                });
            }
            fields.push(InternedValue::String(value));
            if position < bytes.len() && bytes[position] != delimiter {
                return Err(DelimitedError {
                    line,
                    column: position + 1,
                    message: "expected delimiter after quoted field".into(),
                });
            }
        } else {
            let start = position;
            while position < bytes.len() && bytes[position] != delimiter {
                position += 1;
            }
            let value = record[start..position].trim();
            if value.is_empty() {
                return Err(DelimitedError {
                    line,
                    column,
                    message: "empty fields are not values".into(),
                });
            }
            fields.push(match value.parse::<u32>() {
                Ok(number) => InternedValue::U32(number),
                Err(_) => InternedValue::Symbol(value.to_owned()),
            });
        }
        if position == bytes.len() {
            break;
        }
        position += 1;
        if position == bytes.len() {
            return Err(DelimitedError {
                line,
                column: position,
                message: "record ends with an empty field".into(),
            });
        }
    }
    Ok(fields)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelimitedError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl std::fmt::Display for DelimitedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for DelimitedError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_csv_and_tsv_values_in_parallel() {
        let csv = parse_delimited_parallel(
            "from,to\n1,alpha\n2,\"a string\"\n",
            DelimitedOptions {
                delimiter: b',',
                has_header: true,
            },
        )
        .unwrap();
        let tsv = parse_delimited_parallel(
            "alpha\tbeta\n",
            DelimitedOptions {
                delimiter: b'\t',
                has_header: false,
            },
        )
        .unwrap();

        assert_eq!(
            csv[0],
            [InternedValue::U32(1), InternedValue::Symbol("alpha".into())]
        );
        assert_eq!(csv[1][1], InternedValue::String("a string".into()));
        assert_eq!(tsv[0].len(), 2);
    }

    #[test]
    fn parallel_ingestion_preserves_input_order() {
        let source = (0..1_000)
            .map(|value| format!("{value},value-{value}"))
            .collect::<Vec<_>>()
            .join("\n");

        let rows = parse_delimited_parallel(&source, DelimitedOptions::default()).unwrap();

        assert_eq!(rows.len(), 1_000);
        assert_eq!(rows[999][0], InternedValue::U32(999));
    }
}
