#[non_exhaustive]
#[derive(Debug)]
pub enum JoinError {
    UnsupportedJoinKeyType,
    IndexKeyNotComparable,
    MissingIndexRowId,
    HashMemoryLimitExceeded,
    MissingJoinCondition,
    InvalidOrderByColumn,
    UnsupportedJoinType,
    UnsupportedJoinStrategy,
    LeftJoinMissingRightColumns,
}

impl std::fmt::Display for JoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedJoinKeyType => f.write_str("Unsupported join key type"),
            Self::IndexKeyNotComparable => f.write_str("Index key is not comparable to join key"),
            Self::MissingIndexRowId => f.write_str("Index record does not end with a rowid"),
            Self::HashMemoryLimitExceeded => f.write_str("Hash join memory limit exceeded"),
            Self::MissingJoinCondition => f.write_str("Join condition is missing"),
            Self::InvalidOrderByColumn => f.write_str("ORDER BY column is not available"),
            Self::UnsupportedJoinType => f.write_str("Join type is not supported"),
            Self::UnsupportedJoinStrategy => f.write_str("Join strategy is not supported"),
            Self::LeftJoinMissingRightColumns => {
                f.write_str("Left join could not determine right-side column count")
            }
        }
    }
}

impl std::error::Error for JoinError {}
