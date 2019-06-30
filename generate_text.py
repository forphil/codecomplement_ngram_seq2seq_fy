from lstm_model import *
import re

def pick_top_n(preds, vocab_size, top_n=3):
    """
    从预测结果中选取前top_n个最可能的单词,选一个作为下一个单词，增加随机性
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_length, lstm_num_units, prime):
    '''
    生成文本
    :param checkpoint:
    :param n_length:每一个生成的句子最长有多长（不包括已经给出的部分）
    :param lstm_num_units:lstm内部隐层节点数目
    :param prime: 开头部分
    :return:
    '''
    #strings用来存储最后总的代码句
    strings=""
    #store_generate用来保存新生成的一个个词
    store_generate=[]

    #对输入的代码句开头进行处理，每个词存入列表seed_text中
    seed_text = re.split('([: ,.\n(){}\[\]=])', prime)
    seed_text = list(filter(lambda x: x != ' ' and x != '', seed_text))
    last_word=seed_text[-1]

    #将输入的代码头整理成标准格式，先存入strings中
    for k in range(len(seed_text)):
        if seed_text[k] not in mark and last_word not in mark:
            strings += ' ' + seed_text[k]
        else:
            strings += seed_text[k]

    #创建模型
    model = char_RNN(vocab=vocab, n_seqs = 1, n_sequencd_length = 1,
                    lstm_num_units=lstm_num_units, keep_prob=1, num_layers=num_layers,
                 learning_rate=learning_rate, clip_val=5)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        #生成初始状态，为0
        new_state = sess.run(model.init_state)

        #每次输入模型的数的shape都是（1,1），前面的信息靠state传递
        x = np.zeros((1, 1))

        #将已经确定的代码头按单词依次输入模型中
        for i in range(len(seed_text)):
            x[0,0]=vocab2Int[seed_text[i]]
            feed = {model.input: x,
                    model.init_state: new_state}
            preds, new_state = sess.run([model.predtion, model.state],
                                        feed_dict=feed)

        #确定的代码头已经输入结束了，这里的pres和c实际对应第一个生成的新词
        c = pick_top_n(preds, len(vocab))
        #上一行确定了第一个新词，把新词加入store_generate中。c是新词的索引
        store_generate.append(int2Vocab[c])

        #这个循环开始，每一步的输入都是上一步的输出，所有内容都是模型新生成的
        for i in range(n_length):
            #c是上一步预测的单词的索引
            x[0,0]=c
            feed = {model.input: x,
                    model.init_state: new_state}
            preds, new_state = sess.run([model.predtion, model.state],
                                        feed_dict=feed)
            #每一次都从前3个最可能的单词里面选，注意上一个for循环没有这一行，因为上一个for循环每一步的输入都是已经确定的代码头
            c = pick_top_n(preds, len(vocab))
            store_generate.append(int2Vocab[c])
            #n_length=25，最多生成长度为25的代码句，但如果生成了换行符就提前结束
            if int2Vocab[c] == '\n':
                break

        #下面这段代码是把新生成的内容整理成标准格式加到strings后面，思路是看每一个单词前后需不需要加空格
        last_word=seed_text[-1]
        new_word=store_generate[0]
        for k in range(len(store_generate)):
            if new_word not in mark and last_word not in mark:
                strings += ' ' + new_word
            else:
                strings += new_word
            last_word=new_word
            if k!=len(store_generate)-1:
                new_word=store_generate[k+1]

    return strings


mark = '.,()[]:{}\n'        #将后面不需要空格的元素保存在字符串中

if __name__ == '__main__':
    #加载文件
    vocab, vocab2Int, int2Vocab, xs, ys = generated_data()
    #读取checkpoint
    checkpoint = tf.train.latest_checkpoint('checkpoint/')
    print(checkpoint)

    input_strings = input("请输入代码词：")

    #生成文本
    samp = sample(checkpoint, 25, lstm_num_units,  prime=input_strings)
    print('--------------------------------')
    print(samp)
