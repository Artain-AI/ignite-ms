//! Text normalization — strips noise, lowercases, truncates.
//!
//! Reduces vocabulary from millions of unique "words" (URLs, hashes, code)
//! to ~200K real words that hit the vocab cache. Without normalization,
//! cache hit rate drops from 98% to 49%.

use regex::Regex;

/// Configuration for the normalizer.
#[derive(Clone, Debug)]
pub struct NormalizerConfig {
    pub lowercase: bool,
    pub max_chars: usize,
    pub min_chars: usize,
    /// Text prefix prepended after normalization (e.g. "passage: " for e5 models).
    pub prefix: String,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            max_chars: 256,
            min_chars: 10,
            prefix: "passage: ".to_string(),
        }
    }
}

/// Compiled text normalizer. Thread-safe, cloneable (Regex is Clone).
#[derive(Clone)]
pub struct Normalizer {
    config: NormalizerConfig,
    re_markdown_link: Regex,
    re_url: Regex,
    re_email: Regex,
    re_hex_hash: Regex,
    re_hex_prefix: Regex,
    re_number: Regex,
    re_version: Regex,
    re_issue_ref: Regex,
    re_reddit_user: Regex,
    re_reddit_sub: Regex,
    re_file_path: Regex,
    re_repeated_punct: Regex,
    re_whitespace: Regex,
}

impl Normalizer {
    pub fn new(config: NormalizerConfig) -> Self {
        Self {
            config,
            re_markdown_link: Regex::new(r"\[([^\]]*)\]\([^)]*\)").unwrap(),
            re_url: Regex::new(r"(?i)\[?https?://\S+\]?").unwrap(),
            re_email: Regex::new(r"\S+@\S+\.\S+").unwrap(),
            re_hex_hash: Regex::new(r"(?-u:\b)[0-9a-fA-F]{32,}(?-u:\b)").unwrap(),
            re_hex_prefix: Regex::new(r"0x[0-9a-fA-F]+").unwrap(),
            re_number: Regex::new(r"(?-u:\b)\d{6,}(?-u:\b)").unwrap(),
            re_version: Regex::new(r"(?-u:\b)v?\d+\.\d+(?:\.\d+)+(?-u:\b)").unwrap(),
            re_issue_ref: Regex::new(r"#\d+").unwrap(),
            re_reddit_user: Regex::new(r"/?u/\S+").unwrap(),
            re_reddit_sub: Regex::new(r"/?r/\S+").unwrap(),
            re_file_path: Regex::new(r"(?:/[\w.\-]+){2,}").unwrap(),
            re_repeated_punct: Regex::new(r"[!]{3,}|[?]{3,}|[.]{3,}|[,]{3,}|[:]{3,}|[;]{3,}")
                .unwrap(),
            re_whitespace: Regex::new(r"\s+").unwrap(),
        }
    }

    /// Normalize a text string. Returns None if too short after normalization.
    pub fn normalize(&self, text: &str) -> Option<String> {
        let mut s = text.to_string();

        // 1. Strip noise patterns (order matters: markdown/URLs first)
        self.replace_in_place(&mut s, &self.re_markdown_link, "$1");
        self.remove_in_place(&mut s, &self.re_url);
        self.remove_in_place(&mut s, &self.re_email);
        self.remove_in_place(&mut s, &self.re_hex_hash);
        self.remove_in_place(&mut s, &self.re_hex_prefix);
        self.remove_in_place(&mut s, &self.re_file_path);
        self.remove_in_place(&mut s, &self.re_version);
        self.remove_in_place(&mut s, &self.re_issue_ref);
        self.remove_in_place(&mut s, &self.re_reddit_user);
        self.remove_in_place(&mut s, &self.re_reddit_sub);
        self.remove_in_place(&mut s, &self.re_number);

        // 2. Collapse repeated punctuation
        if self.re_repeated_punct.is_match(&s) {
            s = self
                .re_repeated_punct
                .replace_all(&s, |caps: &regex::Captures| {
                    let m = caps.get(0).unwrap().as_str();
                    m.chars().next().unwrap().to_string().repeat(2)
                })
                .into_owned();
        }

        // 3. Normalize whitespace
        if self.re_whitespace.is_match(&s) {
            s = self.re_whitespace.replace_all(&s, " ").into_owned();
        }

        // 4. Lowercase
        let s = if self.config.lowercase {
            s.trim().to_lowercase()
        } else {
            s.trim().to_string()
        };

        // 5. Remove words with no alphabetic characters
        let s: String = s
            .split_whitespace()
            .filter(|w| w.chars().any(|c| c.is_alphabetic()))
            .collect::<Vec<&str>>()
            .join(" ");

        // 6. Truncate
        let s = if self.config.max_chars > 0 && s.len() > self.config.max_chars {
            truncate_utf8(&s, self.config.max_chars).to_string()
        } else {
            s
        };

        // 7. Length check
        if s.len() < self.config.min_chars {
            return None;
        }

        // 8. Prepend model prefix
        let s = if self.config.prefix.is_empty() {
            s
        } else {
            format!("{}{}", self.config.prefix, s)
        };

        Some(s)
    }

    #[inline]
    fn remove_in_place(&self, s: &mut String, re: &Regex) {
        if re.is_match(s) {
            *s = re.replace_all(s, "").into_owned();
        }
    }

    #[inline]
    fn replace_in_place(&self, s: &mut String, re: &Regex, rep: &str) {
        if re.is_match(s) {
            *s = re.replace_all(s, rep).into_owned();
        }
    }
}

fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
