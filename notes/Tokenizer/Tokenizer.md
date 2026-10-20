BasicTokenizer
text
 ↓
UTF-8 bytes
 ↓
BPE



RegexTokenizer
text
 ↓
regex split
 ↓
["Hello", " world", "!"]
 ↓
每个 chunk → UTF-8 bytes
 ↓
每个 chunk 内 BPE