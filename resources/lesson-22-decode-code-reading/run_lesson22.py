#!/usr/bin/env python3
prompt=['BOS','A','B']; generated=[]
cached=len(prompt); pool=prompt+['C']; device=len(pool); max_device=len(prompt)+3
print('after prefill:',pool,'C/D/M=',cached,device,max_device)
for sampled in ['D','EOS']:
    current=pool[cached:device]
    print('\ndecode input:',current,'-> K/V written at positions',list(range(cached,device)))
    cached=device
    device+=1
    pool.append(sampled); generated.append(sampled)
    print('sampled:',sampled,'C/D=',cached,device,'extend=',device-cached,'remain=',max_device-device)
    if sampled=='EOS' or device==max_device:
        print('finish: release pages [0:',cached,'], EOS page was never allocated')
        break
