use core::cmp::Ordering;

use crate::table::ValueRef;

#[inline]
pub(crate) fn compare_value_refs(left: ValueRef<'_>, right: ValueRef<'_>) -> Ordering {
    let rank = |value: ValueRef<'_>| match value {
        ValueRef::Null => 0u8,
        ValueRef::Integer(_) | ValueRef::Real(_) => 1u8,
        ValueRef::Text(_) => 2u8,
        ValueRef::Blob(_) => 3u8,
    };

    let left_rank = rank(left);
    let right_rank = rank(right);
    if left_rank != right_rank {
        return left_rank.cmp(&right_rank);
    }

    match (left, right) {
        (ValueRef::Null, ValueRef::Null) => Ordering::Equal,
        (ValueRef::Integer(l), ValueRef::Integer(r)) => l.cmp(&r),
        (ValueRef::Integer(l), ValueRef::Real(r)) => cmp_f64_total(l as f64, r),
        (ValueRef::Real(l), ValueRef::Integer(r)) => cmp_f64_total(l, r as f64),
        (ValueRef::Real(l), ValueRef::Real(r)) => cmp_f64_total(l, r),
        (ValueRef::Text(l), ValueRef::Text(r)) => l.cmp(r),
        (ValueRef::Blob(l), ValueRef::Blob(r)) => l.cmp(r),
        _ => Ordering::Equal,
    }
}

#[inline]
pub(crate) fn cmp_f64_total(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use super::compare_value_refs;
    use crate::table::ValueRef;

    #[test]
    fn null_orders_before_all_types() {
        assert_eq!(compare_value_refs(ValueRef::Null, ValueRef::Null), Ordering::Equal);
        assert_eq!(compare_value_refs(ValueRef::Null, ValueRef::Integer(0)), Ordering::Less);
        assert_eq!(compare_value_refs(ValueRef::Null, ValueRef::Real(0.0)), Ordering::Less);
        assert_eq!(compare_value_refs(ValueRef::Null, ValueRef::Text(b"a")), Ordering::Less);
        assert_eq!(compare_value_refs(ValueRef::Null, ValueRef::Blob(b"a")), Ordering::Less);
    }

    #[test]
    fn mixed_type_ranking_is_stable() {
        assert_eq!(compare_value_refs(ValueRef::Integer(1), ValueRef::Text(b"a")), Ordering::Less);
        assert_eq!(compare_value_refs(ValueRef::Text(b"a"), ValueRef::Blob(b"a")), Ordering::Less);
        assert_eq!(
            compare_value_refs(ValueRef::Blob(b"a"), ValueRef::Integer(1)),
            Ordering::Greater
        );
    }

    #[test]
    fn mixed_numeric_ordering_uses_total_cmp() {
        assert_eq!(compare_value_refs(ValueRef::Integer(1), ValueRef::Real(2.0)), Ordering::Less);
        assert_eq!(
            compare_value_refs(ValueRef::Real(2.0), ValueRef::Integer(1)),
            Ordering::Greater
        );
        assert_eq!(compare_value_refs(ValueRef::Integer(1), ValueRef::Real(1.0)), Ordering::Equal);
    }

    #[test]
    fn nan_orders_after_numbers() {
        let nan = f64::NAN;
        assert_eq!(compare_value_refs(ValueRef::Real(nan), ValueRef::Real(1.0)), Ordering::Greater);
        assert_eq!(compare_value_refs(ValueRef::Real(1.0), ValueRef::Real(nan)), Ordering::Less);
        assert_eq!(compare_value_refs(ValueRef::Real(nan), ValueRef::Real(nan)), Ordering::Equal);
        assert_eq!(compare_value_refs(ValueRef::Integer(1), ValueRef::Real(nan)), Ordering::Less);
        assert_eq!(
            compare_value_refs(ValueRef::Real(nan), ValueRef::Integer(1)),
            Ordering::Greater
        );
    }
}
