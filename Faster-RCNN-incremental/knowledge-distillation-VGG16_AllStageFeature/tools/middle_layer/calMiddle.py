#对多个余弦相似度进行操作
def mulsimilarToOnesimilar(similarity_lists):
    sum=0;
    index=0;
    for i in range(len(similarity_lists)):
        for j in range(len(similarity_lists[i])):
            similarity_list=similarity_lists[i]
            sum = sum + similarity_list[j]
            index = index + 1
    aveSimilar = sum/index
    return aveSimilar