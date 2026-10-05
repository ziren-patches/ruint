//! 256-bit arithmetic through the Ziren zkVM's precompiles.

use crate::Uint;

unsafe extern "C" {
    /// `x <- (x * y) mod m`, where `m` is the 8 words after `y` and `m = 0`
    /// means `2^256`; `x` and `y` must be 4-byte aligned.
    fn syscall_uint256_mulmod(x: *mut [u32; 8], y: *const [u32; 8]);
}

/// `(a * b) mod m` for 256-bit values, `m = 0` meaning `2^256`.
///
/// The precompile's witness carries the quotient `⌊a · b / m⌋`, which has to
/// stay below the (effective) modulus; `a, b < m` guarantees that, so
/// operands at or above the modulus are reduced before the call
/// ([`Uint::mul_mod`]).  An unreduced pair whose quotient reaches `m` leaves
/// the precompile without a witness, and the shard without a proof.
///
/// The limbs are little-endian 64-bit words, which on this little-endian
/// 32-bit target are the little-endian 32-bit words the precompile reads; the
/// pointers are 8-byte aligned.  The precompile reads `y` and `m` as one
/// 16-word block, so they are laid out contiguously.
#[inline]
pub(crate) fn mul_mod_256<const BITS: usize, const LIMBS: usize>(
    a: Uint<BITS, LIMBS>,
    b: Uint<BITS, LIMBS>,
    m: Uint<BITS, LIMBS>,
) -> Uint<BITS, LIMBS> {
    debug_assert!(BITS == 256 && LIMBS == 4);
    let mut x = *a.as_limbs();
    let mut ym = [0u64; 8];
    ym[..4].copy_from_slice(b.as_limbs());
    ym[4..].copy_from_slice(m.as_limbs());
    unsafe {
        syscall_uint256_mulmod(x.as_mut_ptr().cast(), ym.as_ptr().cast());
    }
    Uint::from_limbs(x)
}

/// `numerator / rhs` and `numerator % rhs` for 256-bit values, left in
/// `numerator` and `rhs`: the quotient and remainder come as a hint computed
/// off the trace, and are checked here.
///
/// With `P = q · b < 2^512`, `L = P mod 2^256` and `S = P mod (2^256 - 1)`,
/// the high half `H = P >> 256` satisfies `S ≡ H + L (mod 2^256 - 1)`, and
/// `H ≤ 2^256 - 2` since `P ≤ (2^256 - 1)^2`; so `S ≡ L` exactly when
/// `H = 0`, i.e. `P = L`.  Then `L + r = a` without carry and `r < b` make
/// `(q, r)` the division of `a` by `b`.
///
/// # Panics
/// Panics if `rhs` is zero or the hint is not the division.
pub(crate) fn div_rem_256<const BITS: usize, const LIMBS: usize>(
    numerator: &mut Uint<BITS, LIMBS>,
    rhs: &mut Uint<BITS, LIMBS>,
) {
    debug_assert!(BITS == 256 && LIMBS == 4);
    let a = *numerator;
    let b = *rhs;
    assert!(!b.is_zero(), "division by zero");
    zkm_lib::unconstrained! {
        let mut q = a;
        let mut r = b;
        crate::algorithms::div::div_inlined(&mut q.limbs, &mut r.limbs);
        let mut bytes = [0u8; 64];
        for i in 0..LIMBS {
            bytes[8 * i..8 * i + 8].copy_from_slice(&q.limbs[i].to_le_bytes());
            bytes[32 + 8 * i..40 + 8 * i].copy_from_slice(&r.limbs[i].to_le_bytes());
        }
        zkm_lib::io::hint_slice(&bytes);
    }
    let hint = zkm_lib::io::read_vec();
    assert_eq!(hint.len(), 64, "division hint length");
    let mut q = Uint::<BITS, LIMBS>::ZERO;
    let mut r = Uint::<BITS, LIMBS>::ZERO;
    for i in 0..LIMBS {
        q.limbs[i] = u64::from_le_bytes(hint[8 * i..8 * i + 8].try_into().expect("8 bytes"));
        r.limbs[i] = u64::from_le_bytes(hint[32 + 8 * i..40 + 8 * i].try_into().expect("8 bytes"));
    }
    assert!(r < b, "division hint remainder");
    let low = mul_mod_256(q, b, Uint::ZERO);
    let ones = Uint::<BITS, LIMBS>::MAX;
    let folded = mul_mod_256(q, b, ones);
    let low_folded = if low == ones { Uint::ZERO } else { low };
    assert!(folded == low_folded, "division hint overflow");
    let (sum, carry) = low.overflowing_add(r);
    assert!(!carry && sum == a, "division hint");
    *numerator = q;
    *rhs = r;
}
