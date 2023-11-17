use acvm::acir::brillig::Opcode as BrilligOpcode;
use acvm::acir::circuit::Circuit;
use acvm::acir::circuit::Opcode;
use acvm::brillig_vm::brillig::BinaryFieldOp;
use acvm::brillig_vm::brillig::Label;
use acvm::brillig_vm::brillig::RegisterIndex;
use rand::distributions::Alphanumeric;
use rand::Rng;
use std::env;
use std::fs;
use std::io::Write;
use std::iter;

use std::collections::HashMap;
use std::path::Path;




/// Module to convert brillig assmebly into powdr assembly

// struct BrilligArchitecture {}

// impl Architecture for BrilligArchitecture {
//     fn instruction_ends_control_flow(instr: &str) -> bool {
//         match instr {
//             "li" | "lui" | "la" | "mv" | "add" | "addi" | "sub" | "neg" | "mul" | "mulhu"
//             | "divu" | "xor" | "xori" | "and" | "andi" | "or" | "ori" | "not" | "slli" | "sll"
//             | "srli" | "srl" | "srai" | "seqz" | "snez" | "slt" | "slti" | "sltu" | "sltiu"
//             | "sgtz" | "beq" | "beqz" | "bgeu" | "bltu" | "blt" | "bge" | "bltz" | "blez"
//             | "bgtz" | "bgez" | "bne" | "bnez" | "jal" | "jalr" | "call" | "ecall" | "ebreak"
//             | "lw" | "lb" | "lbu" | "lh" | "lhu" | "sw" | "sh" | "sb" | "nop" | "fence"
//             | "fence.i" | "amoadd.w.rl" | "amoadd.w" => false,
//             "j" | "jr" | "tail" | "ret" | "trap" => true,
//             _ => {
//                 panic!("Unknown instruction: {instr}");
//             }
//         }
//     }

//     fn get_references<'a, R: asm_utils::ast::Register, F: asm_utils::ast::FunctionOpKind>(
//         instr: &str,
//         args: &'a [asm_utils::ast::Argument<R, F>],
//     ) -> Vec<&'a str> {
//         // fence arguments are not symbols, they are like reserved
//         // keywords affecting the instruction behavior
//         if instr.starts_with("fence") {
//             Vec::new()
//         } else {
//             symbols_in_args(args)
//         }
//     }
// }

fn main() {
    let args: Vec<String> = env::args().collect();
    //dbg!(args);
    // Read in file called bytecode.acir
    let bytecode_path = &args[1];
    let bytecode = fs::read(Path::new(bytecode_path)).expect("Unable to read file");

    let out_asm_path = &args[2];
    // Or read in bytecode itself
    //let bytecode = &args[1];
    // Convert the read-in base64 file into Vec<u8>
    let decoded = base64::decode(&bytecode).expect("Failed to decode base64");
    let bytecode = Vec::from(decoded);

    // Create a new circuit from the bytecode instance
    let circuit: Circuit = Circuit::read(&*bytecode).expect("Failed to deserialize circuit");

    //println!("circuit: {:?}", circuit);

    let avm_bytecode = brillig_to_avm(&circuit.opcodes);
    //fs::write(out_asm_path)
    // Get the brillig opcodes
    //let brillig = extract_brillig(circuit.opcodes);
    //print!("{:?}", brillig);
    let mut file = fs::File::create(out_asm_path).expect("Could not create file");
    println!("Writing avm bytecode (len {0}) to file {1}", avm_bytecode.len(), out_asm_path);
    //file.write_all(base64::encode(&avm_bytecode).as_bytes())
    //    .expect("Could not write to file");
    file.write_all(&avm_bytecode)
        .expect("Could not write to file");

    //let preamble = get_preamble();
    //let program = construct_main(brillig);
    //let powdr = brillig_machine(&preamble, program);

    //println!("powdr: {:?}", powdr);

    // temp write the output to a file
    //let mut file = fs::File::create(out_asm_path).expect("Could not create file");
    //file.write_all(powdr.as_bytes())
    //    .expect("Could not write to file");
}

fn brillig_machine(
    // machines: &[&str],
    preamble: &str,
    // submachines: &[(&str, &str)],
    program: Vec<String>,
) -> String {
    format!(
        r#"
machine Main {{

{}

    function main {{
{}
    }}
}}    
"#,
        preamble,
        program
            .into_iter()
            .map(|line| format!("\t\t{line}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

// Output the powdr assembly with the given circuit
fn construct_main(program: Opcode) -> Vec<String> {
    let mut main_asm: Vec<String> = Vec::new();

    // For each instruction in brillig, we want o
    let trace = match program {
        Opcode::Brillig(brillig) => brillig.bytecode,
        _ => {
            panic!("Opcode is not of type brillig");
        }
    };

    println!("");
    println!("");
    trace.iter().for_each(|i| println!("{:?}", i));
    println!("");
    println!("");

    // Label of [index], String, where index is the generated name of the jump, we will place a jump label there when
    // we encounter it to prove
    let mut index = 0;
    let mut labels: HashMap<Label, String> = HashMap::new();

    for instr in trace {
        println!("{:?}", instr);
        println!("");
        println!("");
        println!("");
        // powdr_asm.push_str(&instr.to_string());

        // If we require a label to be placed at the jump location then we add it
        if let Some(jump) = labels.get(&index) {
            main_asm.push(format!("{}::", jump));
        }

        match instr {
            BrilligOpcode::Const { destination, value } => {
                let number = value.to_usize().to_string();
                main_asm.push(format!("{} <=X= {};", print_register(destination), number));
            }
            BrilligOpcode::Stop => {
                main_asm.push("return;".to_owned());
            }
            BrilligOpcode::Return => {
                main_asm.push("ret;".to_owned());
            }
            // Calls -> look more into how this is performed internally
            // For calls we want to store the current pc in a holding register such that we can return to it
            // We then want to jump to that location in the bytecode
            BrilligOpcode::Call { location } => {
                // Generate a label for the location we are going to
                let label = gen_label();
                labels.insert(location, label.clone()); // This label will be inserted later on!

                main_asm.push(format!("call {};", label));
            }
            BrilligOpcode::BinaryFieldOp {
                destination,
                op,
                lhs,
                rhs,
            } => {
                // Match the given operation
                match op {
                    BinaryFieldOp::Add => {
                        main_asm.push(format!(
                            "{} <== add({}, {});",
                            print_register(destination),
                            print_register(lhs),
                            print_register(rhs)
                        ));
                    }
                    BinaryFieldOp::Sub => {
                        main_asm.push(format!(
                            "{} <== sub({}, {});",
                            print_register(destination),
                            print_register(lhs),
                            print_register(rhs)
                        ));
                    }
                    // Equals is currently a mix of the equals instruction and the using the X is 0 witness column
                    BinaryFieldOp::Equals => {
                        main_asm.push(format!(
                            "tmp <== sub({}, {});",
                            print_register(lhs),
                            print_register(rhs)
                        ));
                        main_asm.push(format!("{} <== eq(tmp);", print_register(destination),));
                    }
                    BinaryFieldOp::Mul => {
                        main_asm.push(format!(
                            "{} <== mul({}, {});",
                            print_register(destination),
                            print_register(lhs),
                            print_register(rhs)
                        ));
                    }
                    // TODO: div
                    _ => println!("not implemented"),
                }
            }
            _ => println!("not implemented"),
        }

        // Increment the index in the instruction array
        index += 1;
    }

    println!("main_asm: {:?}", main_asm);

    main_asm
}

fn gen_label() -> String {
    let mut rng = rand::thread_rng();
    let hex_chars: Vec<char> = "abcdef".chars().collect();
    // Lmao chat gpt fix
    let label: String = iter::repeat(())
        .map(|()| rng.gen_range(0..hex_chars.len()))
        .map(|i| hex_chars[i])
        .take(4)
        .collect();

    label
}

fn print_register(r_index: RegisterIndex) -> String {
    let num = r_index.to_usize();
    format!("r{}", num.to_string()).to_owned()
}

// Read the preamble from the brillig.asm machine
fn get_preamble() -> String {
    let preamble = fs::read_to_string("brillig.asm").expect("Unable to read file");
    preamble
}

fn extract_brillig(opcodes: Vec<Opcode>) -> Opcode {
    if opcodes.len() != 1 {
        panic!("There should only be one brillig opcode");
    }
    let opcode = &opcodes[0];
    if opcode.name() != "brillig" {
        panic!("Opcode is not of type brillig");
    }
    opcode.clone()
}

//fn brillig_to_avm_str(opcodes: &Vec<Opcode>) {
//    if opcodes.len() != 1 {
//        panic!("There should only be one brillig opcode");
//    }
//    let opcode = &opcodes[0];
//    let brillig = match opcode {
//        Opcode::Brillig(brillig) => brillig,
//        _ => panic!("Opcode is not of type brillig"),
//    };
//    //if brillig.name() != "brillig" {
//    //    panic!("Opcode is not of type brillig");
//    //}
//    //opcode.clone()
//    // brillig starts by storing in each calldata entry
//    // into a register starting at register 0 and ending at N-1
//    // where N is inputs.len
//    //println!("\tCALLDATASIZE is {}", brillig.inputs.len());
//    println!("CALLDATACOPY 0 0 0 {}", brillig.inputs.len());
//    // brillig return value(s) start at register N and end at N+M-1
//    // where M is outputs.len
//    //println!("\tRETURNDATASIZE is {}", brillig.outputs.len());
//    //println!("RETURN 0 0 {0} {1}", brillig.inputs.len(), brillig.inputs.len() + brillig.outputs.len());
//
//    for instr in &brillig.bytecode {
//        match instr {
//            BrilligOpcode::BinaryFieldOp { destination, op, lhs, rhs } =>
//                {
//                    let op_name = match op {
//                        BinaryFieldOp::Add => "ADD",
//                        BinaryFieldOp::Sub => "SUB",
//                        BinaryFieldOp::Mul => "MUL",
//                        BinaryFieldOp::Div => "DIV",
//                        BinaryFieldOp::Equals => "EQ",
//                        _ => panic!("Transpiler doesn't know how to process BinaryFieldOp {0}", instr.name()),
//                    };
//                    println!("{0} {1} 0 {2} {3}", op_name, destination.to_usize(), lhs.to_usize(), rhs.to_usize());
//                },
//            BrilligOpcode::Const { destination, value } => println!("SET {0} 0 {1} 0", destination.to_usize(), value.to_usize()),
//            BrilligOpcode::Mov { destination, source } => println!("MOV {0} 0 {1} 0", destination.to_usize(), source.to_usize()),
//            BrilligOpcode::Call { location } => println!("JUMP 0 0 {0} 0", location),
//            BrilligOpcode::Stop {} => println!("RETURN 0 0 0 0"),
//            BrilligOpcode::Return {} => println!("RETURN 0 0 {0} {1}", brillig.inputs.len(), brillig.inputs.len() + brillig.outputs.len()),
//            _ => panic!("Transpiler doesn't know how to process {0} instruction", instr.name()),
//
//        };
//        //print!("Instruction: {}", instr);
//    }
//}

fn brillig_to_avm(opcodes: &Vec<Opcode>) -> Vec<u8> {
    if opcodes.len() != 1 {
        panic!("There should only be one brillig opcode");
    }
    let opcode = &opcodes[0];
    let brillig = match opcode {
        Opcode::Brillig(brillig) => brillig,
        _ => panic!("Opcode is not of type brillig"),
    };
    // brillig starts by storing in each calldata entry
    // into a register starting at register 0 and ending at N-1
    // where N is inputs.len
    //println!("\tCALLDATASIZE is {}", brillig.inputs.len());
    println!("CALLDATACOPY 0 0 0 {}", brillig.inputs.len());
    // brillig return value(s) start at register N and end at N+M-1
    // where M is outputs.len
    //println!("\tRETURNDATASIZE is {}", brillig.outputs.len());
    //println!("RETURN 0 0 {0} {1}", brillig.inputs.len(), brillig.inputs.len() + brillig.outputs.len());
    let mut avm_opcodes = Vec::new();
    // Put calldatasize (which generally is brillig.inputs.len()) to M[0]
    avm_opcodes.push(AVMInstruction {opcode: AVMOpcode::CALLDATASIZE, fields: AVMFields { d0: 0, ..Default::default() }});
    // Put calldata into M[0:calldatasize]
    avm_opcodes.push(AVMInstruction {opcode: AVMOpcode::CALLDATACOPY, fields: AVMFields { d0: 0, s1: 0, ..Default::default() }});

    let pc_offset = brillig.inputs.len();

    for instr in &brillig.bytecode {
        match instr {
            BrilligOpcode::BinaryFieldOp { destination, op, lhs, rhs } => {
                    let op_type = match op {
                        BinaryFieldOp::Add => AVMOpcode::ADD,
                        BinaryFieldOp::Sub => AVMOpcode::SUB,
                        BinaryFieldOp::Mul => AVMOpcode::MUL,
                        BinaryFieldOp::Div => AVMOpcode::DIV,
                        BinaryFieldOp::Equals => AVMOpcode::EQ,
                        _ => panic!("Transpiler doesn't know how to process BinaryFieldOp {0}", instr.name()),
                    };
                    avm_opcodes.push(AVMInstruction {
                        opcode: op_type,
                        fields: AVMFields { d0: destination.to_usize(), s0: lhs.to_usize(), s1: rhs.to_usize(), ..Default::default() }
                    });
                },
            BrilligOpcode::Const { destination, value } =>
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::SET,
                    fields: AVMFields { d0: destination.to_usize(), s0: value.to_usize(), ..Default::default() }
                }),
            BrilligOpcode::Mov { destination, source } =>
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::MOV,
                    fields: AVMFields { d0: destination.to_usize(), s0: source.to_usize(), ..Default::default()}
                }),
            BrilligOpcode::Call { location } =>
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::INTERNALCALL,
                    // +1 for Stop's additional SET opcode
                    fields: AVMFields { s0: *location+pc_offset+1, ..Default::default()}
                    // FIXME: do an initial pass just to update PCs for JUMPs and CALLs
                }),
            BrilligOpcode::Stop {} => {
                //AVMInstruction {opcode: AVMOpcode::RETURN, fields: AVMFields { ..Default::default() }},
                // All that's left is to return. Use an open register/mem-loc for return size.
                // (since all that's left is to return, mem-loc #inputs+#outputs should be free)
                let return_size = brillig.outputs.len();
                let return_size_addr = brillig.inputs.len() + brillig.outputs.len();
                let return_data_addr = brillig.inputs.len(); // return data starts after inputs
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::SET,
                    fields: AVMFields { d0: return_size_addr, s0: return_size, ..Default::default() }
                });
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::RETURN,
                    fields: AVMFields { s0: brillig.inputs.len(), s1: return_size_addr, ..Default::default() }
                });
            },
            BrilligOpcode::Return {} =>
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::INTERNALRETURN,
                    //fields: AVMFields { s0: brillig.inputs.len(), s1: brillig.inputs.len() + brillig.outputs.len(), ..Default::default() }
                    fields: AVMFields { ..Default::default()}
                }),
            _ => panic!("Transpiler doesn't know how to process {0} instruction", instr.name()),

        };
    }
    println!("Printing all AVM instructions!");
    let mut bytecode = Vec::new();
    for avm_instr in avm_opcodes {
        println!("{}", avm_instr.to_string());
        let mut instr_bytes = avm_instr.to_bytes();
        bytecode.append(&mut instr_bytes);
    }
    bytecode
}

pub struct AVMFields {
    d0: usize,
    sd: usize,
    s0: usize,
    s1: usize,
}
impl AVMFields {
    fn to_string(&self) -> String {
        format!("d0: {}, sd: {}, s0: {}, s1: {}", self.d0, self.sd, self.s0, self.s1)
    }
    fn to_bytes(&self) -> Vec<u8> {
        return [
            (self.d0 as u32).to_be_bytes(),
            (self.sd as u32).to_be_bytes(),
            (self.s0 as u32).to_be_bytes(),
            (self.s1 as u32).to_be_bytes(),
        ].concat()
    }
}
impl Default for AVMFields {
    fn default() -> Self {
        AVMFields {
            d0: 0,
            sd: 0,
            s0: 0,
            s1: 0,
        }
    }
}

pub struct AVMInstruction {
    opcode: AVMOpcode,
    fields: AVMFields,
}

impl AVMInstruction {
    fn to_string(&self) -> String {
        format!("opcode: {}, fields: {}", self.opcode.name(), self.fields.to_string())
    }
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        // first byte is opcode
        let op = self.opcode as u8;
        bytes.push(op);
        // convert fields to bytes and append
        let mut fields_bytes = self.fields.to_bytes();
        bytes.append(&mut fields_bytes);
        bytes
    }
}

#[derive(Copy, Clone)]
pub enum AVMOpcode {
  // Arithmetic
  ADD,
  SUB,
  MUL,
  DIV,
  EQ,
  // Memory
  SET,
  MOV,
  CALLDATASIZE,
  CALLDATACOPY,
  // Control flow
  JUMP,
  JUMPI,
  INTERNALCALL,
  INTERNALRETURN,
  // Storage
  SLOAD,
  SSTORE,
  // Contract call control flow
  RETURN,
  CALL,
}
impl AVMOpcode {
    pub fn name(&self) -> &'static str {
        match self {
            AVMOpcode::ADD => "ADD",
            AVMOpcode::SUB => "SUB",
            AVMOpcode::MUL => "MUL",
            AVMOpcode::DIV => "DIV",
            AVMOpcode::EQ => "EQ",
            AVMOpcode::SET => "SET",
            AVMOpcode::MOV => "MOV",
            AVMOpcode::CALLDATASIZE => "CALLDATASIZE",
            AVMOpcode::CALLDATACOPY => "CALLDATACOPY",
            AVMOpcode::JUMP => "JUMP",
            AVMOpcode::JUMPI => "JUMPI",
            AVMOpcode::INTERNALCALL => "INTERNALCALL",
            AVMOpcode::INTERNALRETURN => "INTERNALRETURN",
            AVMOpcode::SLOAD => "SLOAD",
            AVMOpcode::SSTORE => "SSTORE",
            AVMOpcode::RETURN => "RETURN",
            AVMOpcode::CALL => "CALL",
        }
    }
}
