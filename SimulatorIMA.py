# import psutil
# print("so luong nhan", psutil.cpu_count())
import os
import multiprocessing
import time
import argparse

dict_op = {
    'LOAD' :    '0000011',
    'OP_IMM' :  '0010011',
    'JALR' :    '1100111',
    'STORE' :   '0100011',
    'OP' :      '0110011', #LOGIC instruc consist of M Extend and logic ins
    'BRANCH' :  '1100011',
    'AUIPC' :   '0010111',
    'LUI' :     '0110111',
    'JAL' :     '1101111',
    "ATOMIC":   "0101111"
}

def DecimalToBin(Dec, nbit=32):
    if Dec is not None and isinstance(Dec, int):
        if Dec < 0:
            Dec = (1 << nbit) + Dec
            # 4294967296 (100000000000000000000000000000000) + Dec
        bin_str = bin(Dec)[2:].zfill(nbit)
        # Hàm bin() kết quả trả về là chuỗi '0b101'. vì vậy cần lấy từ ký tự 2 để bỏ 0b
        return bin_str
    else:
        raise ValueError("Dec should be a valid integer value.")
# Output: '11111111111111111111111111111011'
# Example:
# print(DecimalToBin(-5))  # Output: '11111111111111111111111111111011'
# dec = 4294967296
#      + (-5)
#      =4294967291 (-5)  Mục đích là để đổi thành số -5

def BinToDecimal(Bin, unsign = 0):
    # tính toán độ dài của chuỗi trừ đi bit dấu
    i = len(Bin) - 1
    if(unsign):
        # nếu unsign là true, chuyển đổi số nhị phân thành số thập phân không dấu
        return int(Bin, 2)
    else:
        # Tính toán giá trị thập phân của số nhị phân bằng cách sử dụng phép chuyển đổi bù hai
        return -2**i + int(Bin[1:], 2) if Bin[0] == '1' else int(Bin, 2)

def multiply_registers(rs1, rs2, opcode):
    result = rs1 * rs2
    if opcode == "MUL":
        return result & 0xFFFFFFFF  # Lấy 32 bit thấp của kết quả
    elif opcode == "MULH":
        return (result >> 32) & 0xFFFFFFFF  # Lấy 32 bit cao của kết quả, đẩy phải 32 bit
    elif opcode == "MULHSU":
        if rs1 < 0:  # Nếu rs1 là số âm
            return ((result >> 32) & 0xFFFFFFFF) | 0xFFFFFFFF00000000  # Lấy 32 bit cao của kết quả, đẩy phải 32 bit và mở rộng dấu
        else:
            return (result >> 32) & 0xFFFFFFFF  # Lấy 32 bit cao của kết quả, đẩy phải 32 bit
    elif opcode == "MULHU":
        return (result >> 32) & 0xFFFFFFFF  # Lấy 32 bit cao của kết quả, đẩy phải 32 bit

def divide_registers(rs1, rs2, opcode):
    if rs2 == 0:
        raise ValueError("Division by zero")
    quotient = rs1 // rs2
    remainder = rs1 % rs2
    if opcode == "DIV":
        return quotient if quotient >= 0 else quotient + 2 ** 32  # Chia có dấu, dương trả về luôn, âm cộng 2^32
    elif opcode == "DIVU":
        return quotient  # Chia không dấu, trả về nguyên
    elif opcode == "REM":
        return remainder if remainder >= 0 else remainder + 2 ** 32  # Phần dư có dấu, dương trả về luôn, âm cộng 2^32
    elif opcode == "REMU":
        return remainder  # Phần dư không dấu, trả về nguyên

def lshift(val, shamt, nbit=32):
    temp = val << shamt
    return temp & ((1 << nbit) - 1)

def rshift(val, shamt, nbit=32):
    sign_bit = 1 << (nbit - 1)
    if val & sign_bit:
        temp = (val >> shamt) | (0xFFFFFFFF << (nbit - shamt))
    else:
        temp = val >> shamt
    return temp

def FieldSplit(ins):
    funct7 = ins[:7]
    rs2 = ins[7:12]
    rs1 = ins[12:17]
    funct3 = ins[17:20]
    rd = ins[20:25]
    opcode = ins[25:32]
    return funct7, rs2, rs1, funct3, rd, opcode

def ReplaceStr(s, new_s, start, end):
    if start == 0:
        return new_s + s[end+1:]
    else:
        return s[:start] + new_s + s[end+1:]
    
def DecoderIme(ins):
    funct7 = imm = rs2 = rs1 = funct3 = rd = opcode = shamt = None
    if ins[-2:] == '11': # 32-bit Ins
        funct7, rs2, rs1, funct3, rd, opcode = FieldSplit(ins)
        rs2 = BinToDecimal(rs2.zfill(32))
        rs1 = BinToDecimal(rs1.zfill(32))
        rd = BinToDecimal(rd.zfill(32))
        if opcode == dict_op['BRANCH']: # Btype
            imm = BinToDecimal(ins[0] + ins[24] + ins[1:7] + ins[20:24] + "0")
        elif opcode in [dict_op["AUIPC"], dict_op["LUI"]]: #U type
            imm = BinToDecimal(ins[:20] + '0'*12) 
        elif opcode == dict_op['JAL']:    #J type
            imm = BinToDecimal(ins[0] + ins[12:20] + ins[11] + ins[1:7] + ins[7:11]+ '0')
        elif opcode in [dict_op['JALR'], dict_op['LOAD']]:    #i type
            imm = BinToDecimal(ins[0:12])
        elif opcode == dict_op['OP_IMM']: #SLLI,SRLI,SRAI
            if(funct3 == '001' or funct3 == '101'):
                imm = BinToDecimal(ins[:7])
                shamt = BinToDecimal(ins[7:12])
            else:   # các lệnh còn lại + lệnh NOP
                imm = BinToDecimal(ins[0:12])
        elif opcode == dict_op['STORE']:
            imm = BinToDecimal(ins[:7] + ins[20:25])
    print("\n",funct7, imm, rs2, rs1, funct3, rd, opcode,shamt)       
    return funct7, imm, rs2, rs1, funct3, rd, opcode,shamt


def atomic_operations(memory_lock, funct3, funct7, rs1, rs2, rd, locked_addresses, REGISTER_FILE, DATAMEM, process_id):
    # print ("gia tri tuyen vao:\t",funct7, rs2, rs1,funct3 , rd)
    addr = REGISTER_FILE[rs1]
    value = REGISTER_FILE[rs2]
    print("\n atomic ", rd, rs2, rs1)
    aq = int(funct7[5])
    rl = int(funct7[6])
    if funct3 == '010':
        if funct7[:5] == '00010':  # LR.W
            print(f"\nProcess {process_id}: addr lock: ", list(locked_addresses))
            if(process_id == 1):
                locked_addresses[rs1] = 1
                REGISTER_FILE[rd] = DATAMEM[addr]
                print(f"\nProcess {process_id}: addr lock: ", list(locked_addresses))
                # Địa chỉ đang bị khóa, chờ đến khi được mở khóa
            else:
                locked_addresses[rs1] = 2
                REGISTER_FILE[rd] = DATAMEM[addr]
                print(f"\nProcess {process_id}: addr lock: ", list(locked_addresses))    
            print(f"\nProcess {process_id}: LR.W X{rd} X{rs1}\n")           
        elif funct7[:5] == '00011':  # SC.W
            print(f"\nProcess {process_id}: SC.W X{rd} x{rs2} X{rs1}\n")
            if locked_addresses[rs1] == process_id:
                REGISTER_FILE[rd] = 0  # success
                locked_addresses[rs1] = 0
                DATAMEM[addr] = value
                print(f"\nProcess {process_id}: addr lock: ", list(locked_addresses))
            else:
                REGISTER_FILE[rd] = 1  # failure
        elif funct7[:5] == '00001':  # AMOSWAP
            temp = DATAMEM[addr]
            DATAMEM[addr] = value
            REGISTER_FILE[rd] = temp
        elif funct7[:5] == '00000':  # AMOADD.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] += value
        elif funct7[:5] == '00100':  # AMOXOR.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] ^= value
        elif funct7[:5] == '01100':  # AMOAND.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] &= value
        elif funct7[:5] == '01000':  # AMOOR.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] |= value
        elif funct7[:5] == '10000':  # AMOMIN.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] = min(DATAMEM[addr], value)
        elif funct7[:5] == '10100':  # AMOMAX.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] = max(DATAMEM[addr], value)
        elif funct7[:5] == '11000':  # AMOMINU.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] = min((DATAMEM[addr]), value)
        elif funct7[:5] == '11100':  # AMOMAXU.W
            REGISTER_FILE[rd] = DATAMEM[addr]
            DATAMEM[addr] = max((DATAMEM[addr]), value)
            
    print(f"\nProcess {process_id}: atomic\n", REGISTER_FILE)
    print(f"\nProcess {process_id}: atomic\n", list(DATAMEM))
            
PC = 0

def Simulator(funct7, rs2, rs1, funct3, rd, imm, opcode, memory_lock,locked_addresses,REGISTER_FILE,DATAMEM, process_id):
    global PC
    print ("gia tri tuyen vao:\t",funct7, imm, rs2, rs1,funct3 , rd, opcode)
    print ("\nTap thanh ghi\n", REGISTER_FILE, PC)
    if opcode == dict_op["LOAD"]:
        PC = PC + 4
        if rd != 0:
           addr_word = (REGISTER_FILE[rs1] + imm) >> 2 # chia cho 4
           if funct3[0] == '0':        # Load signed
                if funct3[1:] == '00':  # Load byte 
                    addr_byte = (REGISTER_FILE[rs1] + imm) % 4
                    temp_mem = DecimalToBin(DATAMEM[addr_word])[(3-addr_byte)*8:(3-addr_byte+1)*8] # Take a byte from memory address
                    temp_mem = temp_mem.rjust(32, temp_mem[0])  # Sign extend to 32bit
                    REGISTER_FILE[rd] = BinToDecimal(temp_mem)
                elif funct3[1:] == '01': # Load half
                    addr_half = int(DecimalToBin((REGISTER_FILE[rs1] + imm) % 4)[30])
                    temp_mem = DecimalToBin(DATAMEM[addr_word])[(1-addr_half)*16:(1-addr_half+1)*16] # Take half word from memory address
                    temp_mem = temp_mem.rjust(32, temp_mem[0])  # Sign extend to 32bit
                    REGISTER_FILE[rd] = BinToDecimal(temp_mem) 
                elif funct3[1:0] == '10': # Load word
                    REGISTER_FILE[rd] = DATAMEM[addr_word]  
           elif funct3[0] == '1':      # Load unsigned
                if funct3[1:] == '00':  # Load byte unsigned
                    addr_word = (REGISTER_FILE[rs1] + imm) >> 2
                    addr_byte = (REGISTER_FILE[rs1] + imm) % 4
                    temp_mem = DecimalToBin(DATAMEM[addr_word])[(3-addr_byte)*8:(3-addr_byte+1)*8] # Take a byte from memory address
                    temp_mem = temp_mem.rjust(32, '0')  # Zero extend to 32bit
                    REGISTER_FILE[rd] = BinToDecimal(temp_mem)
                elif funct3[1:] == '01': # Load half unsigned
                    addr_word = (REGISTER_FILE[rs1] + imm) >> 2
                    addr_half = int(DecimalToBin((REGISTER_FILE[rs1] + imm) % 4)[30])
                    temp_mem = DecimalToBin(DATAMEM[addr_word])[(1-addr_half)*16:(1-addr_half+1)*16] # Take half word from memory address
                    temp_mem = temp_mem.rjust(32, '0')  # Zero extend to 32bit
                    REGISTER_FILE[rd] = BinToDecimal(temp_mem) 
    elif opcode == dict_op['OP_IMM']:
        PC += 4
        if funct3 == '000': #ADDI
           REGISTER_FILE[rd] = REGISTER_FILE[rs1] + imm
           print(REGISTER_FILE[rd])
        elif funct3 == '001':    #SLLI
           #REGISTER_FILE[rd] = lshift(REGISTER_FILE[rs1], shamt)
        #elif funct3 == '010': # SLTI
            if REGISTER_FILE[rs1] < imm:
               REGISTER_FILE[rd] = 1
            else:
                REGISTER_FILE[rd] = 0
        elif funct3 == '011': 
            REGISTER_FILE[rd] = 1 if rs1 < BinToDecimal(DecimalToBin(imm), 1) else 0
            # SET unsign thành 1
        elif funct3 == '100':  
            REGISTER_FILE[rd] = REGISTER_FILE[rs1] ^ imm
        elif funct3 == '110': 
            REGISTER_FILE[rd] = REGISTER_FILE[rs1] | imm 
        elif funct3 == '101'  : # SRLI + SRAI
            #REGISTER_FILE[rd] = rshift(REGISTER_FILE[rs1], shamt)
        #elif funct3 == '111': 
            REGISTER_FILE[rd] = REGISTER_FILE[rs1] & imm 
    elif opcode == dict_op["JAL"]: #JAL stores the address of the instruction following
                                    #the jump (pc+4) into register rd.
        temp = PC + 4
        PC = imm + PC
        if rd != 0: REGISTER_FILE[rd] = temp   
    elif opcode == dict_op["JALR"]:
        temp = PC + 4
        PC = rs1 + imm
        if rd != 0: REGISTER_FILE[rd] = temp    
    elif opcode == dict_op["STORE"]:
        PC = PC + 4
        addr_word = (REGISTER_FILE[rs1] + imm) >> 2
        if funct3 == '000':  # Store byte
            addr_byte = (REGISTER_FILE[rs1] + imm) % 4
            temp_rd = DecimalToBin(REGISTER_FILE[rs2])
            temp_rd = temp_rd[-8:]  # Take only the least significant byte
            temp_mem = DecimalToBin(DATAMEM[addr_word])
            # Replace the byte at the specified position in memory with the new byte value
            DATAMEM[addr_word] = BinToDecimal(ReplaceStr(temp_mem, temp_rd, (3 - addr_byte) * 8, (3 - addr_byte) * 8 + 7))
        elif funct3 == '001':  # Store half
            addr_half = int(DecimalToBin((REGISTER_FILE[rs1] + imm) % 4)[30])
            temp_rd = DecimalToBin(REGISTER_FILE[rs2])
            temp_rd = temp_rd[-16:]  # Take only the least significant half-word
            temp_mem = DecimalToBin(DATAMEM[addr_word])
            # Replace the half-word at the specified position in memory with the new half-word value
            DATAMEM[addr_word] = BinToDecimal(ReplaceStr(temp_mem, temp_rd, (1 - addr_half) * 16, (1 - addr_half) * 16 + 15))
        elif funct3 == '010':  # Store word
            print("toi da vao store\n")
            
            # Directly store the word value into memory
            DATAMEM[addr_word] = REGISTER_FILE[rs2]
            
            print(list(DATAMEM))
    elif opcode == dict_op["AUIPC"]: 
        PC = PC + 4
        if rd != 0: REGISTER_FILE[rd] = PC + imm 
    elif opcode == dict_op["LUI"]:    
        PC = PC + 4
        if rd != 0: REGISTER_FILE[rd] = imm 
    elif opcode == dict_op["BRANCH"]:    
        temp = PC + 4    
        if(funct3 == '000'):
            if(REGISTER_FILE[rs1] == REGISTER_FILE[rs2]):
                PC = PC +imm   
            else:
                PC = temp
        if(funct3 == '001'):
            if(REGISTER_FILE[rs1] != REGISTER_FILE[rs2]):
                PC = PC +imm   
            else:
                PC = temp
        if(funct3 == '100'):
            if(REGISTER_FILE[rs1] < REGISTER_FILE[rs2]):
                PC = PC +imm   
            else:
                PC = temp
        if(funct3 == '100'):
            if(REGISTER_FILE[rs1] >= REGISTER_FILE[rs2]):
                PC = PC +imm   
            else:
                PC = temp
        if(funct3 == '111'): # BGEU
            if(int(DecimalToBin(REGISTER_FILE[rs1])) >= int(DecimalToBin(REGISTER_FILE[rs2]))):
                PC = PC +imm   
            else:
                PC = temp
        if(funct3 == '110'): # BLTU
            if(int(DecimalToBin(REGISTER_FILE[rs1])) < int(DecimalToBin(REGISTER_FILE[rs2]))):
                PC = PC +imm   
            else:
                PC = temp
    elif opcode == dict_op['OP']:
        PC += 4
        if(funct7 == '0000000'):
            if funct3 == '000':
                REGISTER_FILE[rd] = REGISTER_FILE[rs1] + REGISTER_FILE[rs2]
            if funct3 == '001':
                REGISTER_FILE[rd] = REGISTER_FILE[rs1] << REGISTER_FILE[rs2]
            if funct3 == '010':
                REGISTER_FILE[rd] = 1 if REGISTER_FILE[rs1] < REGISTER_FILE[rs2] else 0
            if funct3 == '011': #SLTU
                REGISTER_FILE[rd] = 1 if int(DecimalToBin(REGISTER_FILE[rs1])) < int(DecimalToBin(REGISTER_FILE[rs2])) else 0
            if funct3 == '100':
                REGISTER_FILE[rd] = REGISTER_FILE[rs1] ^ REGISTER_FILE[rs2]
            if funct3 == '101':
                REGISTER_FILE[rd] = REGISTER_FILE[rs1] >> REGISTER_FILE[rs2]
            if funct3 == '110':
                REGISTER_FILE[rd] = REGISTER_FILE[rs1] | REGISTER_FILE[rs2]
            if funct3 == '111':
                REGISTER_FILE[rd] = REGISTER_FILE[rs1] & REGISTER_FILE[rs2]  
        elif(funct7 == '0100000'):   
            if funct3 == '000':
                REGISTER_FILE[rd] = REGISTER_FILE[rs1] - REGISTER_FILE[rs2]  
            if funct3 == '101':    
                REGISTER_FILE[rd] = rshift(REGISTER_FILE[rs1],REGISTER_FILE[rs2]) 
        else: #M Ex
            if(funct3 == '000'):
                REGISTER_FILE[rd] = multiply_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'MUL')       
            if(funct3 == '001'):  
                REGISTER_FILE[rd] = multiply_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'MULH')
            if(funct3 == '010'):  
                REGISTER_FILE[rd] = multiply_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'MULHSU')
            if(funct3 == '011'):  
                REGISTER_FILE[rd] = multiply_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'MULHU')    
            if(funct3 == '100'):
                #REGISTER_FILE[rd] = REGISTER_FILE[rs1] // REGISTER_FILE[rs2]     
                REGISTER_FILE[rd] = divide_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'DIV')
            if(funct3 == '101'):  
                #REGISTER_FILE[rd] = REGISTER_FILE[rs1] // REGISTER_FILE[rs2] 
                REGISTER_FILE[rd] = divide_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'DIVU')
            if(funct3 == '110'):  
                #REGISTER_FILE[rd] = REGISTER_FILE[rs1] % REGISTER_FILE[rs2] 
                REGISTER_FILE[rd] = divide_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'REM')
            if(funct3 == '111'):  
                #REGISTER_FILE[rd] = REGISTER_FILE[rs1] % REGISTER_FILE[rs2]    
                REGISTER_FILE[rd] = divide_registers(REGISTER_FILE[rs1], REGISTER_FILE[rs2], 'REMU')
    elif opcode == dict_op["ATOMIC"]:
        memory_lock.acquire()
        atomic_operations(memory_lock, funct3, funct7, rs1, rs2, rd,locked_addresses, REGISTER_FILE, DATAMEM, process_id)
        PC += 4
        # time.sleep(1)
        memory_lock.release()
    time.sleep(0.01)
    return REGISTER_FILE, DATAMEM, PC


def run_test(binary_strings, REGISTER_FILE, DATAMEM, memory_lock, locked_addresses, process_id, log_file, input_file):
    funct7, imm, rs2, rs1, funct3, rd, opcode, shamt = DecoderIme(binary_strings)
    REGISTER_FILE, DATAMEM, PC = Simulator(funct7, rs2, rs1, funct3, rd, imm, opcode, memory_lock, locked_addresses, REGISTER_FILE, DATAMEM, process_id)
    
    
    expected_register_file_p1 = [0] * 32
    expected_register_file_p2 = [0] * 32
    expected_memory = [0] * 2**8
    expected_register_file_p1[1] = 11
    expected_register_file_p1[2] = 21
    expected_register_file_p1[3] = 0
    expected_register_file_p2[1] = 12
    # expected_register_file_p2[2] = 12
    locked_addresses[31] = 0
    expected_memory[0] = 11
    
    with open(log_file, 'a') as log:
        log.write(f"\nRunning test for input file: {input_file}\n")
        if process_id == 1:
            if REGISTER_FILE == expected_register_file_p1 and list(DATAMEM) == expected_memory:
                log.write("\nTest passed\n")
            else:
                log.write("Test failed\n")
                log.write(f"Expected Register File: {expected_register_file_p1}\n")
                log.write(f"Actual Register File: {REGISTER_FILE}\n")
                log.write(f"Expected Memory: {expected_memory}\n")
                log.write(f"Actual Memory: {list(DATAMEM)}\n")
        else:
            if REGISTER_FILE == expected_register_file_p2 and list(DATAMEM) == expected_memory:
                log.write("\nTest passed\n")
            else:
                log.write("Test failed\n")
                log.write(f"Expected Register File: {expected_register_file_p2}\n")
                log.write(f"Actual Register File: {REGISTER_FILE}\n")
                log.write(f"Expected Memory: {expected_memory}\n")
                log.write(f"Actual Memory: {list(DATAMEM)}\n")


def process_binary_string(binary_strings, REGISTER_FILE, DATAMEM, memory_lock,locked_addresses,process_id):
    for ins in binary_strings:
        funct7, imm, rs2, rs1,funct3 , rd, opcode, shamt = DecoderIme(ins)
        REGISTER_FILE, DATAMEM, PC = Simulator(funct7, rs2, rs1, funct3, rd, imm, opcode, memory_lock,locked_addresses,REGISTER_FILE,DATAMEM, process_id)
        # processed_data.append((REGISTER_FILE, DATAMEM, PC, multiprocessing.current_process().name))
        #  with print_lock:
        #      print(f"{multiprocessing.current_process().name}: {REGISTER_FILE} {PC} \n {DATAMEM}")
    return REGISTER_FILE, DATAMEM, PC

def process_file(input_file,output_file, REGISTER_FILE, DATAMEM, memory_lock,locked_addresses,process_id):
    log_file = os.path.join(output_file, 'log.txt')
    if os.path.isfile(input_file):
        with open(input_file, 'r') as f:
            binary_strings = f.read().splitlines()
            processed_data = process_binary_string(binary_strings, REGISTER_FILE, DATAMEM, memory_lock,locked_addresses, process_id)
            run_test(binary_strings, REGISTER_FILE, DATAMEM, memory_lock, locked_addresses, process_id, log_file, input_file)
        processed_data = [str(item) for item in processed_data]
        processed_data = '\n'.join(processed_data)
        processed_data += '\n'
        
        # with open(output_folder, 'w') as f:
        #     f.write((processed_data))
        #     f.write(str(list(DATAMEM)))
        
        # Create a new output file with the name "o" + input_file name
        output_file_name = "o" + os.path.basename(input_file)
        output_file_path = os.path.join(output_file, output_file_name)
        
        with open(output_file_path, 'w') as f:
            f.write(processed_data)
            f.write(str(list(DATAMEM)))

def process1(input_file_p1, output_file_p1, REGISTER_FILE, DATAMEM,memory_lock,locked_addresses):
    # Your existing code for process_file with slight modification
    # Ensure to pass shared memory objects to functions where necessary
    pid = os.getpid()

    
    # input_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWinvalidatedp1.txt"
    # output_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWinvalidatedp1.txt"
    
    # input_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWnoreverp1.txt"
    # output_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWnoreverp1.txt"
    
    # input_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWsuccessp1.txt"
    # output_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWsuccessp1.txt"
    
    # input_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWoverwritep1.txt"
    # output_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWoverwritep1.txt"
    
    
    # input_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\FileTestInsA.txt"
    # output_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oFileTestInsA.txt"
    process_file(input_file_p1, output_file_p1, REGISTER_FILE, DATAMEM, memory_lock,locked_addresses,process_id=1)

def process2(input_file_p2, output_file_p2, REGISTER_FILE, DATAMEM,memory_lock,locked_addresses):
    # Your existing code for process_file with slight modification
    # Ensure to pass shared memory objects to functions where necessary
    pid = os.getpid()
    
    
    # input_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWinvalidatedp2.txt"
    # output_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWinvalidatedp2.txt"
    
    # input_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWnoreverp2.txt"
    # output_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWnoreverp2.txt"
    
    # input_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWoverwritep2.txt"
    # output_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWoverwritep2.txt"
    
    # input_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\FileTestInsAp2.txt"
    # output_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oFileTestInsAp2.txt"
    process_file(input_file_p2, output_file_p2, REGISTER_FILE, DATAMEM, memory_lock,locked_addresses,process_id=2)

# REGISTER_FILE = [0 for _ in range(32)]
# PC = 0
# DATAMEM = multiprocessing.Array('i', 2**8)

if __name__ == "__main__":
    # Create shared memory objects
    DATA_MEMORY_DEPTH = 2**8
    REGISTER_FILE = [0 for _ in range(32)]
    DATAMEM = multiprocessing.Array('i', DATA_MEMORY_DEPTH)
    memory_lock = multiprocessing.Lock()
    PC = 0
    locked_addresses = multiprocessing.Array('i', 32)

    # Create a lock for printing
    
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i1" , "--input1", help="Input file path for process 1", required=True)
    ap.add_argument("-o1" , "--output1", help="Output file path for process 1", required=True)
    ap.add_argument("-i2" , "--input2", help="Input file path for process 2", required=True)
    ap.add_argument("-o2" , "--output2", help="Output file path for process 2", required=True)
    args = ap.parse_args()

    input_file_p1 = args.input1
    output_file_p1 = args.output1
    input_file_p2 = args.input2
    output_file_p2 = args.output2
    
    # input_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWsuccess2p1.txt"
    # output_file_p1 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWsuccess2dp1.txt"
    # input_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\BinCode\LRWsuccess2p2.txt"
    # output_file_p2 = r"E:\SaveFilePython\Spyder\SimulatorA\output\oLRWsuccess2dp2.txt"
    
    # printing main program process id 
    print("ID of main process: {}".format(os.getpid())) 
    
    start = time.time()
    
    # Create two processes
    p1 = multiprocessing.Process(target=process1, args=(input_file_p1, output_file_p1,REGISTER_FILE, DATAMEM,memory_lock,locked_addresses))
    p2 = multiprocessing.Process(target=process2, args=(input_file_p2, output_file_p2,REGISTER_FILE, DATAMEM,memory_lock,locked_addresses))

    # Start both processes
    p1.start()
    p2.start()

    # Wait for both processes to finish
    p1.join()
    p2.join()
    
    # process IDs 
    print("ID of process p1: {}".format(p1.pid)) 
    print("ID of process p2: {}".format(p2.pid)) 

    
    # print global result list.
    # print result array 
    print("REGISTER_FILE(in main program): {}".format(REGISTER_FILE[:])) 
  
    # print square_sum Value 
    print("DATAMEM(in main program): {}".format(DATAMEM[:])) 
  
    end = time.time()
    # both processes finished 
    print(f"finish time of 2 process paralel: {round(end - start, 2)} second")
    
    # check if processes are alive 
    print("Process p1 is alive: {}".format(p1.is_alive())) 
    print("Process p2 is alive: {}".format(p2.is_alive())) 