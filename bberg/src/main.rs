use acvm::acir::brillig::Opcode as BrilligOpcode;
use acvm::acir::circuit::Circuit;
use acvm::acir::circuit::Opcode;
use acvm::acir::circuit::brillig::Brillig;
use acvm::brillig_vm::brillig::BlackBoxOp;
use acvm::brillig_vm::brillig::RegisterOrMemory;
use acvm::brillig_vm::brillig::{BinaryFieldOp, BinaryIntOp};
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

    let brillig = acir_to_brillig(&circuit.opcodes);
    print_brillig(&brillig);

    let avm_bytecode = brillig_to_avm(&brillig);
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

fn acir_to_brillig(opcodes: &Vec<Opcode>) -> &Brillig {
    if opcodes.len() != 1 {
        panic!("There should only be one brillig opcode");
    }
    let opcode = &opcodes[0];
    let brillig = match opcode {
        Opcode::Brillig(brillig) => brillig,
        _ => panic!("Opcode is not of type brillig"),
    };
    brillig
}

fn print_brillig(brillig: &Brillig) {
    println!("Inputs: {:?}", brillig.inputs);
    for i in 0..brillig.bytecode.len() {//  instr in &brillig.bytecode {
        let instr = &brillig.bytecode[i];
        println!("PC:{0} {1:?}", i, instr);
    }
    println!("Outputs: {:?}", brillig.outputs);
}

const MEMORY_START: usize = 1024;
const POINTER_TO_MEMORY: usize = 2048;
const SCRATCH_START: usize = 2049;

fn brillig_pc_offsets(initial_offset: usize, brillig: &Brillig) -> Vec<usize> {
    // For each instruction that expands to >1 AVM instruction,
    // Construct an array, where each index corresponds to a PC in the original Brillig bytecode.
    // Iterate over the original bytecode, and each time an instruction is encountered that
    // expands to >1 AVM instruction, increase the following(?) entry by the number of added instructions.
    let mut pc_offsets = Vec::new();
    pc_offsets.resize(brillig.bytecode.len(), 0);
    pc_offsets[0] = initial_offset;

    for i in 1..brillig.bytecode.len() {//  instr in &brillig.bytecode {
        let instr = &brillig.bytecode[i];
        let offset = match instr {
            BrilligOpcode::Load {..} => 1,
            BrilligOpcode::Store {..} => 1,
            BrilligOpcode::Stop => 1,
            BrilligOpcode::Trap => 1,
            BrilligOpcode::ForeignCall { function, .. } =>
                match &function[..] {
                    "avm_sload" => 0,
                    "avm_sstore" => 0,
                    "avm_call" => 5,
                    _ => 0,
                },
            BrilligOpcode::BlackBox(bb_op) =>
                match &bb_op {
                    BlackBoxOp::Pedersen {..} => 2,
                    _ => 0,
                },
            _ => 0,
        };
        pc_offsets[i] = pc_offsets[i-1] + offset;
    }
    pc_offsets
}

fn brillig_to_avm(brillig: &Brillig) -> Vec<u8> {
    // brillig starts by storing in each calldata entry
    // into a register starting at register 0 and ending at N-1 where N is inputs.len
    let mut avm_opcodes = Vec::new();
    // Put calldatasize (which generally is brillig.inputs.len()) to M[0]
    avm_opcodes.push(AVMInstruction { opcode: AVMOpcode::CALLDATASIZE, fields: AVMFields { d0: 0, ..Default::default() }});
    // Put calldata into M[0:calldatasize]
    avm_opcodes.push(AVMInstruction { opcode: AVMOpcode::CALLDATACOPY, fields: AVMFields { d0: 0, s1: 0, ..Default::default() }});

    // Put the memory offset into M[MEMORY_START]
    avm_opcodes.push(AVMInstruction { opcode: AVMOpcode::SET, fields: AVMFields { d0: POINTER_TO_MEMORY, s0: MEMORY_START, ..Default::default() }});
    // Put the scratch offset into M[SCRATCH_START]
    //avm_opcodes.push(AVMInstruction { opcode: AVMOpcode::SET, fields: AVMFields { d0: SCRATCH_START, s0: SCRATCH_START, ..Default::default()}})
    let mut next_unused_scratch = SCRATCH_START + 1; // skip 1 because 0th entry is pointer to memory start

    // NOTE: must update this if number of initial instructions ^ pushed above changes
    let pc_offset_for_above_instrs = 3;
    let pc_offsets = brillig_pc_offsets(pc_offset_for_above_instrs, brillig);
    //let pc_offset = 3;

    //let mut next_unused_memory = SCRATCH_MEMORY + 1;

    //for i in 0..brillig.bytecode.len() {//  instr in &brillig.bytecode {
    for instr in &brillig.bytecode {
        //let instr = &brillig.bytecode[i];
        match instr {
            BrilligOpcode::BinaryFieldOp { destination, op, lhs, rhs } => {
                    let op_type = match op {
                        BinaryFieldOp::Add => AVMOpcode::ADD,
                        BinaryFieldOp::Sub => AVMOpcode::SUB,
                        BinaryFieldOp::Mul => AVMOpcode::MUL,
                        BinaryFieldOp::Div => AVMOpcode::DIV,
                        BinaryFieldOp::Equals => AVMOpcode::EQ,
                        // FIXME missing
                        _ => panic!("Transpiler doesn't know how to process BinaryFieldOp {0}", instr.name()),
                    };
                    avm_opcodes.push(AVMInstruction {
                        opcode: op_type,
                        fields: AVMFields { d0: destination.to_usize(), s0: lhs.to_usize(), s1: rhs.to_usize(), ..Default::default() }
                    });
                },
            BrilligOpcode::BinaryIntOp { destination, op, bit_size, lhs, rhs } => {
                // FIXME hacked together
                    let op_type = match op {
                        BinaryIntOp::Add => AVMOpcode::ADD,
                        BinaryIntOp::Sub => AVMOpcode::SUB,
                        BinaryIntOp::Mul => AVMOpcode::MUL,
                        BinaryIntOp::UnsignedDiv => AVMOpcode::DIV,
                        BinaryIntOp::Equals => AVMOpcode::EQ,
                        BinaryIntOp::LessThan => AVMOpcode::LT,
                        BinaryIntOp::LessThanEquals => AVMOpcode::LTE,
                        BinaryIntOp::And => AVMOpcode::AND,
                        BinaryIntOp::Or => AVMOpcode::OR,
                        BinaryIntOp::Xor => AVMOpcode::XOR,
                        BinaryIntOp::Shl => AVMOpcode::SHL,
                        BinaryIntOp::Shr => AVMOpcode::SHR,
                        _ => panic!("Transpiler doesn't know how to process BinaryIntOp {0}", instr.name()),
                    };
                    avm_opcodes.push(AVMInstruction {
                        opcode: op_type,
                        fields: AVMFields { d0: destination.to_usize(), s0: lhs.to_usize(), s1: rhs.to_usize(), ..Default::default() }
                    });
                },
            BrilligOpcode::Jump { location } => {
                let loc_offset = pc_offsets[*location];
                let fixed_loc = *location + loc_offset;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::JUMP,
                    fields: AVMFields { s0: fixed_loc, ..Default::default() }
                });
            },
            BrilligOpcode::JumpIf { condition, location } => {
                let loc_offset = pc_offsets[*location];
                let fixed_loc = *location + loc_offset;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::JUMPI,
                    fields: AVMFields { sd: condition.to_usize(), s0: fixed_loc, ..Default::default() }
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
            BrilligOpcode::Load { destination, source_pointer } => {
                // Brillig Load does R[dst] = M[R[src]]
                // So, we transpile to a MOV with indirect addressing for src (s0)
                // But we offset all Brillig "memory" to avoid collisions with registers since in the AVM everything is memory
                // (via ADD and a scratchpad memory word)
                let src_ptr_after_mem_offset = SCRATCH_START;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::ADD,
                    fields: AVMFields { d0: src_ptr_after_mem_offset, s0: POINTER_TO_MEMORY, s1: source_pointer.to_usize(), ..Default::default()}
                });
                //next_unused_scratch = next_unused_scratch+1;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::MOV,
                    fields: AVMFields { d0: destination.to_usize(), s0: src_ptr_after_mem_offset, s0_indirect: true, ..Default::default()}
                });
            },
            BrilligOpcode::Store { destination_pointer, source } => {
                // Brillig Store does M[R[dst]] = R[src]
                // So, we transpile to a MOV with indirect addressing for dst (d0)
                // But we offset all Brillig "memory" to avoid collisions with registers since in the AVM everything is memory
                // (via ADD and a scratchpad memory word)
                let dst_ptr_after_mem_offset = SCRATCH_START;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::ADD,
                    fields: AVMFields { d0: dst_ptr_after_mem_offset, s0: POINTER_TO_MEMORY, s1: destination_pointer.to_usize(), ..Default::default()}
                });
                //next_unused_scratch = next_unused_scratch+1;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::MOV,
                    fields: AVMFields { d0: dst_ptr_after_mem_offset, s0: source.to_usize(), d0_indirect: true, ..Default::default()}
                });
            },
            BrilligOpcode::Call { location } => {
                let loc_offset = pc_offsets[*location];
                let fixed_loc = *location + loc_offset;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::INTERNALCALL,
                    fields: AVMFields { s0: fixed_loc, ..Default::default()}
                });
            },
            BrilligOpcode::ForeignCall { function, destinations, inputs } => {
                println!("Transpiling ForeignCall::{} with {} destinations and {} inputs", function, destinations.len(), inputs.len());
                match &function[..] {
                    "avm_sload" => {
                        if destinations.len() != 1 || inputs.len() != 1 {
                            panic!("Transpiler expects ForeignCall::{} to have 1 destination and 1 input, got {} and {}", function, destinations.len(), inputs.len());
                        }
                        let slot_operand = match &inputs[0] {
                            RegisterOrMemory::RegisterIndex(index) => index,
                            _ => panic!("Transpiler does not know how to handle ForeignCall::{} with HeapArray/Vector inputs", function),
                        };
                        let dst_operand = match &destinations[0] {
                            RegisterOrMemory::RegisterIndex(index) => index,
                            _ => panic!("Transpiler does not know how to handle ForeignCall::{} with HeapArray/Vector inputs", function),
                        };
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::SLOAD,
                            fields: AVMFields { d0: dst_operand.to_usize(), s0: slot_operand.to_usize(), ..Default::default()}
                        });
                    },
                    "avm_sstore" => {
                        if destinations.len() != 0 || inputs.len() != 2 {
                            panic!("Transpiler expects ForeignCall::{} to have 0 destinations and 2 inputs, got {} and {}", function, destinations.len(), inputs.len());
                        }
                        let slot_operand = match &inputs[0] {
                            RegisterOrMemory::RegisterIndex(index) => index,
                            _ => panic!("Transpiler does not know how to handle ForeignCall::{} with HeapArray/Vector inputs", function),
                        };
                        let value_operand = match &inputs[1] {
                            RegisterOrMemory::RegisterIndex(index) => index,
                            _ => panic!("Transpiler does not know how to handle ForeignCall::{} with HeapArray/Vector inputs", function),
                        };
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::SSTORE,
                            fields: AVMFields { d0: slot_operand.to_usize(), s0: value_operand.to_usize(), ..Default::default()}
                        });
                    },
                    "avm_call" => {
                        if destinations.len() != 1 || inputs.len() != 3 {
                            panic!("Transpiler expects ForeignCall::{} to have 1 destinations and 3 inputs, got {} and {}", function, destinations.len(), inputs.len());
                        }
                        let gas_operand = match &inputs[0] {
                            RegisterOrMemory::RegisterIndex(index) => index,
                            _ => panic!("Transpiler does not know how to handle ForeignCall::{} with HeapArray/Vector for gas operand", function),
                        };
                        let target_address_operand = match &inputs[1] {
                            RegisterOrMemory::RegisterIndex(index) => index,
                            _ => panic!("Transpiler does not know how to handle ForeignCall::{} with HeapArray/Vector for target_address operand", function),
                        };
                        let args_heap_array = match &inputs[2] {
                            RegisterOrMemory::HeapArray(heap_array) => heap_array,
                            _ => panic!("Transpiler expects ForeignCall::{}'s inputs[2] to be a HeapArray for a call's args", function),
                        };
                        let return_heap_array= match &destinations[0] {
                            RegisterOrMemory::HeapArray(heap_array) => heap_array,
                            // TODO: if heap array, need to generate RETURNDATASIZE and RETURNDATACOPY instructions as size isn't know ahead of time!
                            // Note that when return data is in a HeapVector, it lives in dest1, and dest0 is a register.... Not sure what for.
                            //RegisterOrMemory::HeapVector(heap_vec) => heap_vec,
                            _ => panic!("Transpiler expects ForeignCall::{}'s destination[0] to be a HeapArray for a call's return data", function),
                        };
                        // Construct a block of memory in the scratchpad that will be pointed to by argsAndRetOffset:
                        // 0th entry: argsOffset (address of args_heap_array in memory)
                        // 1st entry: argsSize (size of args_heap_array)
                        // 2nd entry: retOffset (address of return_heap_array in memory)
                        // 3nd entry: retSize (size of return_heap_array)
                        let mem_containing_args_offset = SCRATCH_START;
                        let mem_containing_args_size = SCRATCH_START+1;
                        let mem_containing_ret_offset = SCRATCH_START+2;
                        let mem_containing_ret_size = SCRATCH_START+3;
                        // Pointer to the above block ^
                        let mem_containing_args_and_ret_offset = SCRATCH_START+4;

                        // offset inputs into region dedicated to brillig "memory"
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::ADD,
                            fields: AVMFields { d0: mem_containing_args_offset, s0: POINTER_TO_MEMORY, s1: args_heap_array.pointer.to_usize(), ..Default::default()}
                        });
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::SET,
                            fields: AVMFields { d0: mem_containing_args_size, s0: args_heap_array.size, ..Default::default()}
                        });
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::ADD,
                            fields: AVMFields { d0: mem_containing_ret_offset, s0: POINTER_TO_MEMORY, s1: return_heap_array.pointer.to_usize(), ..Default::default()}
                        });
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::SET,
                            // TODO use heap array and size instead of 1
                            //fields: AVMFields { d0: mem_containing_ret_size, s0: 1, ..Default::default()}
                            fields: AVMFields { d0: mem_containing_ret_size, s0: return_heap_array.size, ..Default::default()}
                        });

                        //if destinations.len() > 1 {
                        //    match &destinations[1] {
                        //        RegisterOrMemory::HeapVector(return_heap_vec) => {
                        //            avm_opcodes.push(AVMInstruction {
                        //                opcode: AVMOpcode::ADD,
                        //                fields: AVMFields { d0: mem_containing_ret_offset, s0: POINTER_TO_MEMORY, s1: return_heap_vec.pointer.to_usize(), ..Default::default()}
                        //            });
                        //            avm_opcodes.push(AVMInstruction {
                        //                opcode: AVMOpcode::MOV,
                        //                fields: AVMFields { d0: mem_containing_ret_size, s0: return_heap_vec.size.to_usize(), ..Default::default()}
                        //            });
                        //        },
                        //        _ => panic!("Transpiler expects ForeignCall::{}'s destination[1] to be a HeapVector for return data", function),
                        //    };
                        //}

                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::SET,
                            fields: AVMFields { d0: mem_containing_args_and_ret_offset, s0: SCRATCH_START, ..Default::default()}
                        });

                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::CALL,
                            fields: AVMFields { s0: gas_operand.to_usize(), s1: target_address_operand.to_usize(), sd: mem_containing_args_and_ret_offset, ..Default::default()}
                        });
                    },
                    _ => panic!("Transpiler does not recognize ForeignCall function {0}", function),
                }
            },
            BrilligOpcode::Stop {} => {
                let return_size = brillig.outputs.len();
                //// Use the register right after inputs and outputs as a scratch register for return size
                //let return_size_addr = brillig.inputs.len() + brillig.outputs.len();
                let mem_for_return_size = SCRATCH_START;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::SET,
                    fields: AVMFields { d0: mem_for_return_size, s0: return_size, ..Default::default() }
                });
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::RETURN,
                    // TODO: is outputs[0] (start of returndata) always "2"? Seems so....
                    fields: AVMFields { s0: 2, s1: mem_for_return_size, ..Default::default() }
                });
            },
            BrilligOpcode::Trap {} => {
                // Trap is a revert, but for now it does not support a return value
                let return_size = brillig.outputs.len();
                //// Use the register right after inputs and outputs as a scratch register for return size
                //let return_size_addr = brillig.inputs.len() + brillig.outputs.len();
                let mem_for_return_size = SCRATCH_START;
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::SET,
                    fields: AVMFields { d0: mem_for_return_size, s0: return_size, ..Default::default() }
                });
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::REVERT,
                    // TODO: is outputs[0] (start of returndata) always "2"? Seems so....
                    fields: AVMFields { s0: 2, s1: mem_for_return_size, ..Default::default() }
                });
            },
            BrilligOpcode::Return {} =>
                avm_opcodes.push(AVMInstruction {
                    opcode: AVMOpcode::INTERNALRETURN,
                    fields: AVMFields { ..Default::default()}
                }),
            BrilligOpcode::BlackBox(bb_op) =>
                match &bb_op {
                    BlackBoxOp::Pedersen { inputs, domain_separator, output } => {
                        let mem_containing_args_offset = SCRATCH_START;
                        let mem_containing_ret_offset = SCRATCH_START+1;

                        // offset inputs into region dedicated to brillig "memory"
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::ADD,
                            fields: AVMFields { d0: mem_containing_args_offset, s0: POINTER_TO_MEMORY, s1: inputs.pointer.to_usize(), ..Default::default()}
                        });
                        // offset output into region dedicated to brillig "memory"
                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::ADD,
                            fields: AVMFields { d0: mem_containing_ret_offset, s0: POINTER_TO_MEMORY, s1: output.pointer.to_usize(), ..Default::default()}
                        });

                        avm_opcodes.push(AVMInstruction {
                            opcode: AVMOpcode::PEDERSEN,
                            fields: AVMFields { d0: mem_containing_ret_offset, sd: inputs.size.to_usize(), s0: domain_separator.to_usize(), s1: mem_containing_args_offset, ..Default::default() }
                        });
                        // pedersen always has output size 2, so that can be hardcoded in simulator
                    },
                    _ => panic!("Transpiler doesn't know how to process BlackBoxOp::{:?} instruction", bb_op),
                },
                //avm_opcodes.push(AVMInstruction {
                //    opcode: AVMOpcode::INTERNALRETURN,
                //    fields: AVMFields { ..Default::default()}
                //}),
            _ => panic!("Transpiler doesn't know how to process {} instruction", instr.name()),

        };
    }
    // TODO: separate function for printing avm instructions
    // TODO: separate function for converting avm instruction vec to bytecode
    println!("Printing all AVM instructions!");
    let mut bytecode = Vec::new();
    for i in 0..avm_opcodes.len() {
        let avm_instr = &avm_opcodes[i];
        println!("PC:{0}: {1}", i, avm_instr.to_string());
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
    d0_indirect: bool,
    s0_indirect: bool,
}
impl AVMFields {
    fn to_string(&self) -> String {
        format!("d0: {}, sd: {}, s0: {}, s1: {}, d0_indirect: {}, s0_indirect: {}", self.d0, self.sd, self.s0, self.s1, self.d0_indirect, self.s0_indirect)
    }
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = [
            (self.d0 as u32).to_be_bytes(),
            (self.sd as u32).to_be_bytes(),
            (self.s0 as u32).to_be_bytes(),
            (self.s1 as u32).to_be_bytes(),
        ].concat();
        bytes.push(self.d0_indirect as u8);
        bytes.push(self.s0_indirect as u8);
        bytes
    }
}
impl Default for AVMFields {
    fn default() -> Self {
        AVMFields {
            d0: 0,
            sd: 0,
            s0: 0,
            s1: 0,
            d0_indirect: false,
            s0_indirect: false,
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
  LT,
  LTE,
  AND,
  OR,
  XOR,
  NOT,
  SHL,
  SHR,
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
  REVERT,
  CALL,
  // Blackbox ops
  PEDERSEN,
}
impl AVMOpcode {
    pub fn name(&self) -> &'static str {
        match self {
            AVMOpcode::ADD => "ADD",
            AVMOpcode::SUB => "SUB",
            AVMOpcode::MUL => "MUL",
            AVMOpcode::DIV => "DIV",
            AVMOpcode::EQ => "EQ",
            AVMOpcode::LT => "LT",
            AVMOpcode::LTE => "LTE",
            AVMOpcode::AND => "AND",
            AVMOpcode::OR => "OR",
            AVMOpcode::XOR => "XOR",
            AVMOpcode::NOT => "NOT",
            AVMOpcode::SHL => "SHL",
            AVMOpcode::SHR => "SHR",

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
            AVMOpcode::REVERT => "REVERT",
            AVMOpcode::CALL => "CALL",

            AVMOpcode::PEDERSEN => "PEDERSEN",
        }
    }
}
