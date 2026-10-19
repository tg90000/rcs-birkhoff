//! RCS: exact minimum Birkhoff decompositions of doubly stochastic matrices with finite precision.
//!
//! The problem asks for a convex combination of as few permutation matrices as possible that equals a given doubly
//! stochastic matrix A. The entries of A are multiples of 1/s, and we work with the scaled matrix D = sA, whose entries
//! are integers and whose row and column sums are all equal to s. A decomposition writes D as a sum of terms
//! c_l P_l with integer weights c_l, that is, with weights of precision 1/s.
//!
//! RCS has three parts (Sect. 3 of the article), and a run (`solver`) uses them in two chains:
//! * Recover (`frequency`, `fixing`, `largest_first`): finds the weights of matrices built from few permutation
//!   matrices and places them on the positions of D;
//! * Certify (`bounds`, `blocks`, `sat`): the degree, the two-line bound and SAT certificates, the bound chain;
//! * Search (`birkhoff_algorithm`, `beam`, `lns`, `sat`): the Birkhoff algorithm with the zeroing rule and the
//!   logarithmic tie-break rule, a beam search over zeroing steps, and improvement methods.
//!
//! `matching` and `threshold` hold the matching code of the Birkhoff algorithm (Sect. 4 of the article), and `cli` the
//! command-line options of the binaries.

pub mod beam;
pub mod birkhoff_algorithm;
pub mod blocks;
pub mod bounds;
pub mod cli;
pub mod decomposition;
pub mod fixing;
pub mod frequency;
pub mod instance;
pub mod largest_first;
pub mod lns;
pub mod matching;
pub mod reference;
pub mod sat;
pub mod solver;
pub mod threshold;
