use ast::analyzed::Analyzed;
use std::{io, str::FromStr};

use number::{BigInt, Bn254Field, DegreeType, FieldElement};

use crate::circuit_builder::analyzed_to_cpp;

/// Barretenberg codegen
///
/// This module will take pil compiler output and make it generate relation header files that can be compiled into bberg

pub struct BBergCodegen {
    // Note: Im not sure we need to know the degree ahead of time
    // degree: DegreeType,
}

impl BBergCodegen {
    pub fn new(_degree: DegreeType) -> Self {
        Self {}
    }

    pub fn new_from_setup(input: &mut impl io::Read) -> Result<Self, io::Error> {
        println!("warning bberg: new_from_setup not implemented");
        Ok(Self {})
    }

    // Note: only returns vec<u8> to keep with the interface
    pub fn build_ast<F: FieldElement>(
        &self,
        pil: &Analyzed<F>,
        fixed: &[(&str, Vec<F>)],
        witness: &[(&str, Vec<F>)],
    ) -> Vec<u8> {
        let circuit = analyzed_to_cpp(pil, fixed, witness);

        Vec::new()
    }

    pub fn assert_field_is_compatible<F: FieldElement>() {
        if Bn254Field::modulus().to_arbitrary_integer() != F::modulus().to_arbitrary_integer() {
            panic!("powdr modulus doesn't match halo2 modulus. Make sure you are using Bn254");
        }
    }
}

// static str& template = r#"

// "#;
