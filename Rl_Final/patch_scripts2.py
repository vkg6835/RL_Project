import os
import re

files = [
    "music_emotion_ql.py",
    "music_emotion_sarsa.py",
    "music_emotion_double_ql.py",
    "music_emotion_double_sarsa.py"
]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    # 6. leave_one_subject_out print
    # Remove conv_sym
    content = re.sub(r'\s*conv_sym = "v" if res\["exact_converged"\] else "x"', '', content)
    
    # Replace the print block
    old_print = r'print\(f"\s*S\{test_idx\+1:02d\} \| err=\{res\[\'final_err\'\]:5\.1f\} deg \| "\s*\n\s*f"exact=\{conv_sym\} \| <30 deg=\{\'v\' if res\[\'conv_30\'\] else \'x\'\} \| "\s*\n\s*f"<60 deg=\{\'v\' if res\[\'conv_60\'\] else \'x\'\} \| "\s*\n\s*f"start=\{res\[\'start\'\]:12s\} \| "\s*\n\s*f"\{\' -> \'\.join\(res\[\'emotions\'\]\)\}"\)'
    
    new_print = '''print(f"    S{test_idx+1:02d} | err={res['final_err']:5.1f} deg | "
              f"<=20 deg={'v' if res['conv_20'] else 'x'} | "
              f"start={res['start']:12s} | "
              f"{' -> '.join(res['emotions'])}")'''
              
    content = re.sub(old_print, new_print, content, flags=re.MULTILINE | re.DOTALL)
                     
    with open(f, "w") as file:
        file.write(content)

print("Patching complete.")
