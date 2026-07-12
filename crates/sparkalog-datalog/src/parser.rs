use crate::{
    SourceAtom, SourceLiteral, SourceProgram, SourceRule, SourceTerm, SourceValue, Span, Spanned,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
}

impl Diagnostic {
    pub fn line_column(&self, source: &str) -> (usize, usize) {
        let prefix = &source[..self.span.start.min(source.len())];
        let line = prefix.bytes().filter(|&byte| byte == b'\n').count() + 1;
        let column = prefix
            .rsplit_once('\n')
            .map_or(prefix.len(), |(_, tail)| tail.len())
            + 1;
        (line, column)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseOutput {
    pub program: SourceProgram,
    pub diagnostics: Vec<Diagnostic>,
}

impl ParseOutput {
    pub fn is_success(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

pub fn parse_program(source: &str) -> ParseOutput {
    let (tokens, mut diagnostics) = lex(source);
    let mut parser = Parser {
        tokens,
        position: 0,
        diagnostics: Vec::new(),
    };
    let program = parser.parse_program();
    diagnostics.extend(parser.diagnostics);
    ParseOutput {
        program,
        diagnostics,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenKind {
    Ident(String),
    Symbol(String),
    String(String),
    Number(u32),
    LeftParen,
    RightParen,
    Comma,
    Dot,
    ColonDash,
    Bang,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    kind: TokenKind,
    span: Span,
}

fn lex(source: &str) -> (Vec<Token>, Vec<Diagnostic>) {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut diagnostics = Vec::new();
    let mut position = 0;
    while position < bytes.len() {
        let start = position;
        match bytes[position] {
            byte if byte.is_ascii_whitespace() => position += 1,
            b'#' | b'%' => {
                position += 1;
                while position < bytes.len() && bytes[position] != b'\n' {
                    position += 1;
                }
            }
            b'/' if bytes.get(position + 1) == Some(&b'/') => {
                position += 2;
                while position < bytes.len() && bytes[position] != b'\n' {
                    position += 1;
                }
            }
            b'(' => push_punctuation(&mut tokens, &mut position, TokenKind::LeftParen),
            b')' => push_punctuation(&mut tokens, &mut position, TokenKind::RightParen),
            b',' => push_punctuation(&mut tokens, &mut position, TokenKind::Comma),
            b'.' => push_punctuation(&mut tokens, &mut position, TokenKind::Dot),
            b'!' => push_punctuation(&mut tokens, &mut position, TokenKind::Bang),
            b':' if bytes.get(position + 1) == Some(&b'-') => {
                position += 2;
                tokens.push(Token {
                    kind: TokenKind::ColonDash,
                    span: Span::new(start, position),
                });
            }
            b'\'' => {
                position += 1;
                let value_start = position;
                while position < bytes.len() && is_identifier_continue(bytes[position]) {
                    position += 1;
                }
                if value_start == position {
                    diagnostics.push(Diagnostic {
                        message: "expected a symbol name after apostrophe".into(),
                        span: Span::new(start, position),
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Symbol(source[value_start..position].to_owned()),
                        span: Span::new(start, position),
                    });
                }
            }
            b'"' => lex_string(source, &mut position, &mut tokens, &mut diagnostics),
            byte if byte.is_ascii_digit() => {
                position += 1;
                while position < bytes.len() && bytes[position].is_ascii_digit() {
                    position += 1;
                }
                match source[start..position].parse::<u32>() {
                    Ok(value) => tokens.push(Token {
                        kind: TokenKind::Number(value),
                        span: Span::new(start, position),
                    }),
                    Err(_) => diagnostics.push(Diagnostic {
                        message: "integer literal does not fit in u32".into(),
                        span: Span::new(start, position),
                    }),
                }
                if source[start..position].parse::<u32>().is_err() {
                    tokens.push(Token {
                        kind: TokenKind::Invalid,
                        span: Span::new(start, position),
                    });
                }
            }
            byte if is_identifier_start(byte) => {
                position += 1;
                while position < bytes.len() && is_identifier_continue(bytes[position]) {
                    position += 1;
                }
                tokens.push(Token {
                    kind: TokenKind::Ident(source[start..position].to_owned()),
                    span: Span::new(start, position),
                });
            }
            _ => {
                position += 1;
                diagnostics.push(Diagnostic {
                    message: format!("unexpected character {:?}", &source[start..position]),
                    span: Span::new(start, position),
                });
                tokens.push(Token {
                    kind: TokenKind::Invalid,
                    span: Span::new(start, position),
                });
            }
        }
    }
    (tokens, diagnostics)
}

fn push_punctuation(tokens: &mut Vec<Token>, position: &mut usize, kind: TokenKind) {
    let start = *position;
    *position += 1;
    tokens.push(Token {
        kind,
        span: Span::new(start, *position),
    });
}

fn lex_string(
    source: &str,
    position: &mut usize,
    tokens: &mut Vec<Token>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let bytes = source.as_bytes();
    let start = *position;
    *position += 1;
    let mut value = String::new();
    let mut terminated = false;
    while *position < bytes.len() {
        match bytes[*position] {
            b'"' => {
                *position += 1;
                terminated = true;
                break;
            }
            b'\\' => {
                *position += 1;
                let Some(&escaped) = bytes.get(*position) else {
                    break;
                };
                let character = match escaped {
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'"' => '"',
                    b'\\' => '\\',
                    _ => {
                        diagnostics.push(Diagnostic {
                            message: "unsupported string escape".into(),
                            span: Span::new(*position - 1, *position + 1),
                        });
                        escaped as char
                    }
                };
                value.push(character);
                *position += 1;
            }
            _ => {
                let remainder = &source[*position..];
                let character = remainder.chars().next().expect("source is not exhausted");
                value.push(character);
                *position += character.len_utf8();
            }
        }
    }
    if terminated {
        tokens.push(Token {
            kind: TokenKind::String(value),
            span: Span::new(start, *position),
        });
    } else {
        diagnostics.push(Diagnostic {
            message: "unterminated string literal".into(),
            span: Span::new(start, *position),
        });
    }
}

fn is_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_identifier_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
    diagnostics: Vec<Diagnostic>,
}

impl Parser {
    fn parse_program(&mut self) -> SourceProgram {
        let mut program = SourceProgram::default();
        while !self.is_finished() {
            let result = if self.at_output_directive() {
                self.parse_output()
                    .map(|output| program.outputs.push(output))
            } else {
                self.parse_rule().map(|rule| program.rules.push(rule))
            };
            if let Err(diagnostic) = result {
                self.diagnostics.push(diagnostic);
                self.synchronize();
            }
        }
        program
    }

    fn parse_rule(&mut self) -> Result<SourceRule, Diagnostic> {
        let head = self.parse_atom()?;
        let mut body = Vec::new();
        if self
            .consume_if(|kind| matches!(kind, TokenKind::ColonDash))
            .is_some()
        {
            loop {
                body.push(self.parse_literal()?);
                if self
                    .consume_if(|kind| matches!(kind, TokenKind::Comma))
                    .is_none()
                {
                    break;
                }
            }
        }
        let dot = self.expect(
            |kind| matches!(kind, TokenKind::Dot),
            "expected `.` after rule",
        )?;
        Ok(SourceRule {
            span: head.span.join(dot.span),
            head,
            body,
        })
    }

    fn parse_literal(&mut self) -> Result<SourceLiteral, Diagnostic> {
        let bang = self.consume_if(|kind| matches!(kind, TokenKind::Bang));
        let atom = self.parse_atom()?;
        Ok(SourceLiteral {
            negated: bang.is_some(),
            span: bang.map_or(atom.span, |token| token.span.join(atom.span)),
            atom,
        })
    }

    fn parse_atom(&mut self) -> Result<SourceAtom, Diagnostic> {
        let predicate = self.expect_ident("expected a predicate name")?;
        self.expect(
            |kind| matches!(kind, TokenKind::LeftParen),
            "expected `(` after predicate name",
        )?;
        let mut terms = Vec::new();
        if !self.check(|kind| matches!(kind, TokenKind::RightParen)) {
            loop {
                terms.push(self.parse_term()?);
                if self
                    .consume_if(|kind| matches!(kind, TokenKind::Comma))
                    .is_none()
                {
                    break;
                }
            }
        }
        let right = self.expect(
            |kind| matches!(kind, TokenKind::RightParen),
            "expected `)` after atom",
        )?;
        let span = predicate.span.join(right.span);
        Ok(SourceAtom {
            predicate,
            terms,
            span,
        })
    }

    fn parse_term(&mut self) -> Result<Spanned<SourceTerm>, Diagnostic> {
        let Some(token) = self.advance() else {
            return Err(self.error_here("expected a variable or constant"));
        };
        let term = match token.kind {
            TokenKind::Ident(name) => SourceTerm::Variable(name),
            TokenKind::Symbol(value) => SourceTerm::Constant(SourceValue::Symbol(value)),
            TokenKind::String(value) => SourceTerm::Constant(SourceValue::String(value)),
            TokenKind::Number(value) => SourceTerm::Constant(SourceValue::U32(value)),
            _ => {
                return Err(Diagnostic {
                    message: "expected a variable or constant".into(),
                    span: token.span,
                });
            }
        };
        Ok(Spanned::new(term, token.span))
    }

    fn parse_output(&mut self) -> Result<Spanned<String>, Diagnostic> {
        self.expect(|kind| matches!(kind, TokenKind::Dot), "expected `.output`")?;
        let directive = self.expect_ident("expected `output` after `.`")?;
        if directive.value != "output" {
            return Err(Diagnostic {
                message: format!("unknown directive .{}", directive.value),
                span: directive.span,
            });
        }
        let predicate = self.expect_ident("expected predicate name after `.output`")?;
        self.consume_if(|kind| matches!(kind, TokenKind::Dot));
        Ok(predicate)
    }

    fn at_output_directive(&self) -> bool {
        self.check(|kind| matches!(kind, TokenKind::Dot))
    }

    fn expect_ident(&mut self, message: &str) -> Result<Spanned<String>, Diagnostic> {
        let Some(token) = self.advance() else {
            return Err(self.error_here(message));
        };
        match token.kind {
            TokenKind::Ident(value) => Ok(Spanned::new(value, token.span)),
            _ => Err(Diagnostic {
                message: message.into(),
                span: token.span,
            }),
        }
    }

    fn expect(
        &mut self,
        predicate: impl FnOnce(&TokenKind) -> bool,
        message: &str,
    ) -> Result<Token, Diagnostic> {
        let Some(token) = self.advance() else {
            return Err(self.error_here(message));
        };
        if predicate(&token.kind) {
            Ok(token)
        } else {
            Err(Diagnostic {
                message: message.into(),
                span: token.span,
            })
        }
    }

    fn consume_if(&mut self, predicate: impl FnOnce(&TokenKind) -> bool) -> Option<Token> {
        if self.peek().is_some_and(|token| predicate(&token.kind)) {
            self.advance()
        } else {
            None
        }
    }

    fn check(&self, predicate: impl FnOnce(&TokenKind) -> bool) -> bool {
        self.peek().is_some_and(|token| predicate(&token.kind))
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        self.position += usize::from(token.is_some());
        token
    }

    fn is_finished(&self) -> bool {
        self.position >= self.tokens.len()
    }

    fn error_here(&self, message: &str) -> Diagnostic {
        let position = self.peek().map_or_else(
            || self.tokens.last().map_or(0, |token| token.span.end),
            |token| token.span.start,
        );
        Diagnostic {
            message: message.into(),
            span: Span::new(position, position),
        }
    }

    fn synchronize(&mut self) {
        while let Some(token) = self.advance() {
            if matches!(token.kind, TokenKind::Dot) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_facts_rules_negation_constants_and_outputs() {
        let source = r#"
            edge('a, "b").
            weight('a, 42).
            path(x, y) :- edge(x, y), !blocked(y).
            .output path
        "#;
        let parsed = parse_program(source);

        assert_eq!(parsed.diagnostics, []);
        assert_eq!(parsed.program.rules.len(), 3);
        assert!(parsed.program.rules[0].is_fact());
        assert_eq!(parsed.program.rules[2].body.len(), 2);
        assert!(parsed.program.rules[2].body[1].negated);
        assert_eq!(parsed.program.outputs[0].value, "path");
        assert_eq!(
            parsed.program.to_string(),
            "edge('a, \"b\").\nweight('a, 42).\npath(x, y) :- edge(x, y), !blocked(y).\n.output path\n"
        );
    }

    #[test]
    fn recovers_at_statement_boundaries() {
        let source = "broken(x :- nope. good('value).";
        let parsed = parse_program(source);

        assert_eq!(parsed.diagnostics.len(), 1);
        assert_eq!(parsed.program.rules.len(), 1);
        assert_eq!(parsed.program.rules[0].head.predicate.value, "good");
        assert_eq!(parsed.diagnostics[0].line_column(source), (1, 10));
    }

    #[test]
    fn reports_lexical_errors_without_losing_valid_rules() {
        let parsed = parse_program("bad(4294967296). ok(1). @");

        assert!(parsed.diagnostics.len() >= 2);
        assert_eq!(parsed.program.rules.len(), 1);
        assert_eq!(parsed.program.rules[0].head.predicate.value, "ok");
    }
}
