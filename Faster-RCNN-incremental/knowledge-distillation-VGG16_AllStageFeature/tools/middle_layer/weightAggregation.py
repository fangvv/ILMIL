#对教师和学生的权重进行聚合
import torch
import numpy as np
def teacherAddStudent(student, teacher):
    fh = open('qian.txt', 'w', encoding='utf-8')
    fh.write(str(student))
    fh.close()
    with torch.no_grad():
        for sname, sparam in student.items():
            for tname, tparam in teacher.items():
                if (sname == tname) and (sname == "module.rpn.head.cls_logits.weight"):
                    student[sname] = torch.cat((tparam, sparam[40:80,:,:,:]),0)
                    break;
                elif (sname == tname) and (sname == "module.rpn.head.cls_logits.bias"):
                    student[sname] = torch.cat((tparam, sparam[40:80]),0)
                    break;
                #if (sname == tname) and (sname != "module.rpn.head.cls_logits.weight") and (sname != "module.rpn.head.cls_logits.bias"):
                elif sname == tname:
                    #print ("更改权值")
                    student[sname] = torch.add(sparam*0.3, tparam*0.7)
                    #student.items.sparam.copy_(torch.div(sparam, 2))
                    #student[sname] = torch.div(sparam, 2)
                    break;
    fh = open('hou.txt', 'w', encoding='utf-8')
    fh.write(str(student))
    fh.close()
    return student