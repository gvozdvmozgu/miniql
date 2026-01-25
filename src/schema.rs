#[derive(Clone, Debug)]
pub(crate) struct TableSchema {
    pub(crate) columns: Vec<String>,
    pub(crate) unique_indexes: Vec<Vec<String>>,
    pub(crate) without_rowid: bool,
}

#[derive(Clone, Debug)]
struct ColumnConstraintInfo {
    name: String,
    unique_index: bool,
}

pub(crate) fn parse_table_schema(sql: &str) -> TableSchema {
    let without_rowid = contains_token_sequence(sql, &["WITHOUT", "ROWID"]);
    let Some(inner) = extract_parenthesized(sql) else {
        return TableSchema { columns: Vec::new(), unique_indexes: Vec::new(), without_rowid };
    };
    let mut columns = Vec::new();
    let mut unique_indexes = Vec::new();
    for part in split_top_level(inner) {
        if is_table_constraint(part) {
            if let Some(cols) = parse_table_constraint_index_cols(part) {
                unique_indexes.push(cols);
            }
            continue;
        }
        if let Some(info) = parse_column_def(part) {
            columns.push(info.name.clone());
            if info.unique_index {
                unique_indexes.push(vec![info.name]);
            }
        }
    }
    TableSchema { columns, unique_indexes, without_rowid }
}

pub(crate) fn parse_index_columns(sql: &str) -> Option<Vec<String>> {
    if sql.to_ascii_uppercase().contains(" WHERE ") {
        return None;
    }

    let paren_start = find_on_paren(sql)?;
    let inner = extract_parenthesized_at(sql, paren_start)?;
    let mut cols = Vec::new();
    for part in split_top_level(inner) {
        let name = parse_identifier(part)?;
        cols.push(name.to_ascii_lowercase());
    }
    Some(cols)
}

fn parse_table_constraint_index_cols(part: &str) -> Option<Vec<String>> {
    let has_primary = contains_token_sequence(part, &["PRIMARY", "KEY"]);
    let has_unique = contains_token(part, "UNIQUE");
    if !has_primary && !has_unique {
        return None;
    }
    let inner = extract_parenthesized(part)?;
    let mut cols = Vec::new();
    for item in split_top_level(inner) {
        let name = parse_identifier(item)?;
        cols.push(name.to_ascii_lowercase());
    }
    if cols.is_empty() { None } else { Some(cols) }
}

fn is_table_constraint(part: &str) -> bool {
    let word = first_word(part);
    matches!(
        word.as_deref(),
        Some("CONSTRAINT") | Some("PRIMARY") | Some("UNIQUE") | Some("CHECK") | Some("FOREIGN")
    )
}

fn first_word(part: &str) -> Option<String> {
    let bytes = part.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    if matches!(bytes[i], b'"' | b'`' | b'[' | b'\'') {
        return None;
    }
    let start = i;
    while i < bytes.len() && is_ident_char(bytes[i]) {
        i += 1;
    }
    if start == i {
        return None;
    }
    Some(part[start..i].to_ascii_uppercase())
}

fn parse_column_def(part: &str) -> Option<ColumnConstraintInfo> {
    let (name, end) = parse_identifier_span(part)?;
    let name = name.to_ascii_lowercase();
    let rest = part[end..].trim_start();
    let (type_name, rest) = parse_optional_type(rest);
    let has_primary = contains_token_sequence(rest, &["PRIMARY", "KEY"]);
    let has_unique = contains_token(rest, "UNIQUE");
    let integer_primary = type_name.as_deref() == Some("INTEGER") && has_primary;
    let unique_index = (has_primary || has_unique) && !integer_primary;
    Some(ColumnConstraintInfo { name, unique_index })
}

fn parse_optional_type(rest: &str) -> (Option<String>, &str) {
    let Some((token, end)) = parse_identifier_span(rest) else {
        return (None, rest);
    };
    let upper = token.to_ascii_uppercase();
    if is_constraint_keyword(&upper) {
        return (None, rest);
    }
    (Some(upper), rest[end..].trim_start())
}

fn is_constraint_keyword(token: &str) -> bool {
    matches!(
        token,
        "CONSTRAINT"
            | "PRIMARY"
            | "UNIQUE"
            | "NOT"
            | "NULL"
            | "CHECK"
            | "DEFAULT"
            | "COLLATE"
            | "REFERENCES"
            | "GENERATED"
            | "AS"
            | "STORED"
            | "VIRTUAL"
            | "ON"
            | "AUTOINCREMENT"
    )
}

fn parse_identifier(part: &str) -> Option<String> {
    parse_identifier_span(part).map(|(token, _)| token)
}

fn parse_identifier_span(part: &str) -> Option<(String, usize)> {
    let bytes = part.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    let first = bytes[i];
    if first == b'(' {
        return None;
    }
    if first == b'"' || first == b'`' || first == b'[' {
        let (token, end) = parse_quoted(bytes, i)?;
        let token = strip_qualifier(&token);
        return Some((token, end));
    }
    let start = i;
    while i < bytes.len()
        && !bytes[i].is_ascii_whitespace()
        && !matches!(bytes[i], b'(' | b',' | b')')
    {
        i += 1;
    }
    if start == i {
        return None;
    }
    let token = &part[start..i];
    let token = strip_qualifier(token);
    Some((token.to_owned(), i))
}

fn parse_quoted(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    let quote = bytes[start];
    let end_quote = if quote == b'[' { b']' } else { quote };
    let mut i = start + 1;
    let mut out = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b == end_quote {
            if end_quote == b'"' && i + 1 < bytes.len() && bytes[i + 1] == end_quote {
                out.push(end_quote);
                i += 2;
                continue;
            }
            return Some((String::from_utf8_lossy(&out).into_owned(), i + 1));
        }
        out.push(b);
        i += 1;
    }
    None
}

fn strip_qualifier(name: &str) -> String {
    match name.rsplit_once('.') {
        Some((_prefix, suffix)) => suffix.to_owned(),
        None => name.to_owned(),
    }
}

fn split_top_level(input: &str) -> Vec<&str> {
    let bytes = input.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth = 0u32;
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut in_bracket = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_backtick {
            if b == b'`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }
        if in_bracket {
            if b == b']' {
                in_bracket = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'`' => in_backtick = true,
            b'[' => in_bracket = true,
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
            }
            b',' if depth == 0 => {
                parts.push(input[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < input.len() {
        parts.push(input[start..].trim());
    }
    parts
}

fn contains_token(input: &str, token: &str) -> bool {
    contains_token_sequence(input, &[token])
}

fn contains_token_sequence(input: &str, seq: &[&str]) -> bool {
    if seq.is_empty() {
        return true;
    }
    let tokens = tokens_upper(input);
    if tokens.len() < seq.len() {
        return false;
    }
    for i in 0..=tokens.len() - seq.len() {
        if seq.iter().enumerate().all(|(j, s)| tokens[i + j] == *s) {
            return true;
        }
    }
    false
}

fn tokens_upper(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i]) {
                i += 1;
            }
            tokens.push(input[start..i].to_ascii_uppercase());
            continue;
        }
        i += 1;
    }
    tokens
}

fn extract_parenthesized(sql: &str) -> Option<&str> {
    let bytes = sql.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            return extract_parenthesized_at(sql, i);
        }
        i += 1;
    }
    None
}

fn extract_parenthesized_at(sql: &str, start: usize) -> Option<&str> {
    let bytes = sql.as_bytes();
    if start >= bytes.len() || bytes[start] != b'(' {
        return None;
    }
    let mut depth = 0u32;
    let mut i = start;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut in_bracket = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_backtick {
            if b == b'`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }
        if in_bracket {
            if b == b']' {
                in_bracket = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'`' => in_backtick = true,
            b'[' => in_bracket = true,
            b'(' => {
                depth += 1;
                if depth == 1 {
                    i += 1;
                    continue;
                }
            }
            b')' => {
                if depth == 1 {
                    return Some(&sql[start + 1..i]);
                }
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn find_on_paren(sql: &str) -> Option<usize> {
    let bytes = sql.as_bytes();
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut in_backtick = false;
    let mut in_bracket = false;
    let mut seen_on = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_single {
            if b == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                in_single = false;
            }
            i += 1;
            continue;
        }
        if in_double {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                in_double = false;
            }
            i += 1;
            continue;
        }
        if in_backtick {
            if b == b'`' {
                in_backtick = false;
            }
            i += 1;
            continue;
        }
        if in_bracket {
            if b == b']' {
                in_bracket = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'\'' => in_single = true,
            b'"' => in_double = true,
            b'`' => in_backtick = true,
            b'[' => in_bracket = true,
            b'(' if seen_on => return Some(i),
            _ => {}
        }

        if !seen_on && is_ident_start(b) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i]) {
                i += 1;
            }
            if sql[start..i].eq_ignore_ascii_case("ON") {
                let prev = start.checked_sub(1).and_then(|p| bytes.get(p).copied());
                let next = bytes.get(i).copied();
                if prev.is_none_or(|c| !is_ident_char(c)) && next.is_none_or(|c| !is_ident_char(c))
                {
                    seen_on = true;
                }
            }
            continue;
        }

        i += 1;
    }
    None
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
