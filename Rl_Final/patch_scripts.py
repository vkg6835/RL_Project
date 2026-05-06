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
    
    # 1. LAMBDA
    content = re.sub(r'LAMBDA\s*=\s*-100\.0', 'LAMBDA                 = -150.0', content)
    
    # 2. Reward
    content = re.sub(r'r\s*=\s*1\.0\s*if\s*\(abs', 'r = 10.0 if (abs', content)
    
    # 3. evaluate_subject return dict
    content = re.sub(r'"exact_converged":.*?,', '', content)
    content = re.sub(r'"conv_30":.*?,', '', content)
    content = re.sub(r'"conv_60":.*?,', '"conv_20":         fe <= 20.0,', content)
    content = re.sub(r'"converged":.*?,', '', content)
    
    # 4. leave_one_subject_out signature
    content = re.sub(r'use_reward_shaping:\s*bool\s*=\s*True\)\s*->\s*list:', 
                     'use_reward_shaping: bool = True,\n                          start_emotion: str = None) -> list:', content)
                     
    # 5. leave_one_subject_out evaluate_subject
    content = re.sub(r'res\s*=\s*evaluate_subject\(agent,\s*test_subject,\s*bins=bins\)',
                     'res            = evaluate_subject(agent, test_subject, bins=bins, start_emotion=start_emotion)', content)
                     
    # 6. leave_one_subject_out print
    print_regex = r'conv_sym\s*=\s*.*?\n.*?deg=.*?\n.*?start=\{res\[\'start\'].*?\n.*?emotions\'\]\}\)"\)'
    new_print = '''print(f"    S{test_idx+1:02d} | err={res['final_err']:5.1f} deg | "
              f"<=20 deg={'v' if res['conv_20'] else 'x'} | "
              f"start={res['start']:12s} | "
              f"{' -> '.join(res['emotions'])}")'''
    content = re.sub(print_regex, new_print, content, flags=re.DOTALL)
    
    # 7. grid_search signature
    content = re.sub(r'use_reward_shaping:\s*bool\s*=\s*True\)\s*->\s*dict:',
                     'use_reward_shaping: bool = True,\n                start_emotion: str = None) -> dict:', content)
                     
    # 8. grid_search evaluate_subject
    content = re.sub(r'evaluate_subject\(agent,\s*subjects\[vi\],\s*bins=bins\)\["final_err"\]',
                     'evaluate_subject(agent, subjects[vi], bins=bins, start_emotion=start_emotion)["final_err"]', content)
                     
    with open(f, "w") as file:
        file.write(content)

print("Patching complete.")
