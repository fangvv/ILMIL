import torch
def model_conv():
    #teacher模型只用改名字
    list1 = [ "middle_extractor1.0.weight", "middle_extractor1.0.bias",
              "middle_extractor1.2.weight", "middle_extractor1.2.bias",
              "middle_extractor2.0.weight", "middle_extractor2.0.bias",
              "middle_extractor2.2.weight", "middle_extractor2.2.bias",
              "middle_extractor3.0.weight", "middle_extractor3.0.bias",
              "middle_extractor3.2.weight", "middle_extractor3.2.bias",
              "middle_extractor3.4.weight", "middle_extractor3.4.bias",
              "middle_extractor4.0.weight", "middle_extractor4.0.bias",
              "middle_extractor4.2.weight", "middle_extractor4.2.bias",
              "middle_extractor4.4.weight", "middle_extractor4.4.bias",
              "extractor.0.weight", "extractor.0.bias",
              "extractor.2.weight", "extractor.2.bias",
              "extractor.4.weight", "extractor.4.bias"]
    list2 = ["extractor.0.weight", "extractor.0.bias",
             "extractor.2.weight", "extractor.2.bias",
             "extractor.5.weight", "extractor.5.bias",
             "extractor.7.weight", "extractor.7.bias",
             "extractor.10.weight", "extractor.10.bias",
             "extractor.12.weight", "extractor.12.bias",
             "extractor.14.weight", "extractor.14.bias",
             "extractor.17.weight", "extractor.17.bias",
             "extractor.19.weight", "extractor.19.bias",
             "extractor.21.weight", "extractor.21.bias",
             "extractor.24.weight", "extractor.24.bias",
             "extractor.26.weight", "extractor.26.bias",
             "extractor.28.weight", "extractor.28.bias", ]
    model = torch.load("../pretrained_model/fasterrcnn_12231419_16_0.7089819173484984", map_location='cpu')
    for i in range(len(list1)):
        model["model"][list1[i]] = model["model"].pop(list2[i])
    torch.save(model, "../pretrained_model/fasterrcnn_12231419_16_0.7089819173484984_te_su.pth")


    # 学生模型
    # list1 = ["middle_extractor1.0.weight", "middle_extractor1.0.bias",
    #          "middle_extractor1.2.weight", "middle_extractor1.2.bias",
    #          "middle_extractor2.0.weight", "middle_extractor2.0.bias",
    #          "middle_extractor2.2.weight", "middle_extractor2.2.bias",
    #          "middle_extractor3.0.weight", "middle_extractor3.0.bias",
    #          "middle_extractor3.2.weight", "middle_extractor3.2.bias",
    #          "middle_extractor3.4.weight", "middle_extractor3.4.bias",
    #          "middle_extractor4.0.weight", "middle_extractor4.0.bias",
    #          "middle_extractor4.2.weight", "middle_extractor4.2.bias",
    #          "middle_extractor4.4.weight", "middle_extractor4.4.bias",
    #          "extractor.0.weight", "extractor.0.bias",
    #          "extractor.2.weight", "extractor.2.bias",
    #          "extractor.4.weight", "extractor.4.bias"]
    # list2 = ["extractor.0.weight", "extractor.0.bias",
    #          "extractor.2.weight", "extractor.2.bias",
    #          "extractor.5.weight", "extractor.5.bias",
    #          "extractor.7.weight", "extractor.7.bias",
    #          "extractor.10.weight", "extractor.10.bias",
    #          "extractor.12.weight", "extractor.12.bias",
    #          "extractor.14.weight", "extractor.14.bias",
    #          "extractor.17.weight", "extractor.17.bias",
    #          "extractor.19.weight", "extractor.19.bias",
    #          "extractor.21.weight", "extractor.21.bias",
    #          "extractor.24.weight", "extractor.24.bias",
    #          "extractor.26.weight", "extractor.26.bias",
    #          "extractor.28.weight", "extractor.28.bias", ]
    # model = torch.load("../pretrained_model/fasterrcnn_11231335_0.570649955391841", map_location='cpu')
    # for i in range(len(list1)):
    #     model["model"][list1[i]] = model["model"].pop(list2[i])
    # model["model"].pop('head.score.weight')
    # model["model"].pop('head.score.bias')
    # model["model"].pop('head.cls_loc.weight')
    # model["model"].pop('head.cls_loc.bias')
    # model["model"]['head.score.weight'] = torch.nn.init.normal_(torch.rand(21,4096))
    # model["model"]['head.score.bias'] = torch.nn.init.normal_(torch.rand(21))
    # model["model"]['head.cls_loc.weight'] = torch.nn.init.normal_(torch.rand(84, 4096))
    # model["model"]['head.cls_loc.bias'] = torch.nn.init.normal_(torch.rand(84))
    # torch.save(model, "../pretrained_model/fasterrcnn_11231335_0.570649955391841_student.pth")

if __name__ == '__main__':
    model_conv()