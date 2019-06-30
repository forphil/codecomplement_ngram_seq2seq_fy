#coding=utf-8
'''
结构：
generated_data方法，预处理语料，返回xs和ys，shape分别为（62600，50），（62600，1）
get_a_batch方法，将generated_data的生成的xs，ys处理成一个个batch，用yielld生成

model_input方法，定义了输入和标签的格式，占位符，返回这两个
model_lstm方法，定义了双向LSTMcell，返回cell和初始值
model_output方法，定义了LSTM输出后的参数进行全连接和softmax，返回全连接后的结果和经过softmax后的结果
model_loss方法，计算实际标签和预测标签之前的loss，返回loss
model_optimizer方法，对所有可训练的变量实施梯度剪裁

class char_RNN中，集合了上述5个方法，定义了计算节点

主函数中，新建了char_RNN类，将数据喂入类中，执行计算节点，得到loss，保存模型
'''

import tensorflow as tf
import numpy as np
import os
import re
import collections

#每个batch有100个句子
n_seqs=100
# 提取50个代码词组成的序列
n_sequencd_length=50
#每5个代码词采样一个新序列
step=5
#LSTM单元的隐藏层参数数量，下一步要进softmax
lstm_num_units=256
#LSTM有几层，2就是双向LSTM
num_layers=2
learning_rate=0.003
#dropout保留的比例
keep_prob=0.7
sentences = []  # 保存所提取的序列
next_words = []  # 保存目标代码词

def build_dict(text, min_word_freq=50):
    word_freq = collections.defaultdict(int)  # 定义一个int型的词频词典，并提供默认值
    for w in text:  # 遍历列表中的元素，元素出现一次，频次加一
        word_freq[w] += 1
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items())          # filter将词频数量低于指定值的代码词删除。
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))         # key用于指定排序的元素，因为sorted默认使用list中每个item的第一个元素从小到大排列，所以这里通过lambda进行前后元素调序，并对词频去相反数，从而将词频最大的排列在最前面
    words, _ = list(zip(*word_freq_sorted))         #获取每一个代码词
    return words


def generated_data():
    filename = open('data/data0502.txt', 'r', encoding='utf-8')  # 打开数据文件

    text = filename.read()  # 将数据读取到字符串text中
    text = ' '.join(re.split(' |\t|\v', text))  # 将数据中的空格符统一，便于后期处理(原始数据中空格符包含\t、\v等)
    text = re.split('([: ,.\n(){}\[\]=])', text)  # 将字符串数据按照括号中的符号进行分割，分割成列表格式，并且在列表中保留分隔符

    text = list(filter(lambda x: x != ' ' and x != '', text))  # 将列表中的空格和非空格筛选掉
    list_text = text  # 保留一份列表格式的数据

    cut_words = list_text  # 将列表形式的元数据保存在cut_words中
    for i in range(0, len(cut_words) - n_sequencd_length, step):
        sentences.append(cut_words[i:i + n_sequencd_length])  # 将元数据按照步长来存储在每个序列中
        next_words.append(cut_words[i + n_sequencd_length])  # 将目标代码词存储在next_words中

    words = list(build_dict(list_text, 0))  # 创建代码词词典，返回的是一个不含重复的代码词词典，不包含词频。

    #词到索引的dict
    vocab2Int = {word: index for index, word in enumerate(words)}
    #未登录词
    vocab2Int['<unk>'] = len(words)
    #索引到词的dict
    int2Vocab = {index: word for word, index in vocab2Int.items()}

    xs = np.zeros((len(sentences), n_sequencd_length))  # 初始化x
    ys = np.zeros((len(sentences), 1))  # 初始化y
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            xs[i, t] = vocab2Int[word]  # 将代码词转换成向量形式的编码
        ys[i, 0] = vocab2Int[next_words[i]]

    # 一共能凑多少个batch
    batch_num = xs.shape[0] // n_seqs
    # 扔掉凑不齐一个batch的句子，处理前有62659行，处理后有62600行
    xs = xs[0:batch_num * n_seqs, :]
    ys = ys[0:batch_num * n_seqs, :]

    return words, vocab2Int, int2Vocab, xs, ys


#输入是所有训练数据，每次yield一个batch的数据
def get_a_batch(xs,ys,n_seqs):
    for i in range(0,xs.shape[0],n_seqs):
        x=xs[i:i+n_seqs,:]
        y = ys[i:i + n_seqs, :]
        yield x,y


def model_input(n_seqs, n_sequencd_length):
    '''
    模型输入部分
    :param n_seqs:每次输入的样本数目，就是batch_size
    :param n_sequencd_length: 每个样本的长度
    :return:
    '''
    #初始化两个的占位
    #输入的大小为样本数目*样本长度
    #因为每个字符会对应一个字符的输出，所以target与input大小一致
    input = tf.placeholder(dtype=tf.int32, shape=(n_seqs, n_sequencd_length), name='input')
    target = tf.placeholder(dtype=tf.int32, shape=(n_seqs, 1), name='target')

    return input, target


def model_lstm(lstm_num_units, keep_prob, num_layers, n_seqs):
    '''
    构建lstm
    :param lstm_num_units:每个lstm节点内部的隐层节点数目【【这个大小是LSTM单元的输出维度，是charRNN那张图第三层output的维度】】
    :param keep_prob: drop比例
    :param num_layers: 层数目
    :param n_seqs: 每次传入多少样本
    :return:
    '''

    #创建列表，后续生成的节点都会放在列表里
    lstms = []

    #循环创建层，num_layers为2，就是双向LSTM
    for i in range(num_layers):
        #单独创建一层lstm节点
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units)
        #添加drop 【【经过这一步，实际上drop已经代表了LSTM单元了，表示一个带有dropout的LSTM单元】】
        drop = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        #将节点放入list中
        lstms.append(drop)

    #创建lstm
    cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
    #初始化输入状态，0
    init_state = cell.zero_state(n_seqs, dtype=tf.float32)
    return cell, init_state

def model_output(lstm_output, in_size, out_size):
    '''
    模型输出
    lstm输出后在这里再通过softmax运算

    因为在输入时输入矩阵是n_seqs * n_sequencd_length大小的，lstm的隐层节点数目是lstm_num_units，
    所以lstm输出的大小是[n_seqs, n_sequencd_length, lstm_num_units]
    Softmax的大小是词汇表的长度，也就是len(vocab)
    在输出中一共有n_seqs * n_sequencd_length个字符，然后我们需要转换成[n_seqs * n_sequencd_length, lstm_num_units]，也就是每一个字符对应一个lstm_num_units维度的向量，准备下一步softmax
    再通过softmax层，softmax接收的参数维度是lstm_num_units，所以中间的w大小应该是[lstm_num_units, len(vocab)]

    首先需要做的是讲lstm的输出的维度转换成[n_seqs * n_sequencd_length, lstm_num_units]，变成二维的，再进行后续处理
    :param lstm_output: lstm的输出， 在这里是len(vocab)
    :param in_size: lstm的输出大小，在这里是lstm_num_units
    :param out_size:这里是softmax输出的维度，有多少个 字母+标点，size就是多少
    :return:
    '''

    #第二维只保留最后一个输出，也就是通过滑动窗口预测下一个单词，这里的最后一个输出就是下一个单词
    lstm_output=lstm_output[:,-1,:]
    # 将维度转换为[n_seqs * n_sequencd_length, lstm_num_units]
    lstm_output = tf.reshape(lstm_output, shape=(-1, in_size))

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal(shape=(in_size, out_size), stddev=0.1), dtype=tf.float32, name='softmax_w')
        softmax_b = tf.Variable(tf.zeros(shape=(out_size)), dtype=tf.float32, name='softmax_b')

    #计算输出，为了计算loss，logits也要作为返回值
    logits = tf.matmul(lstm_output, softmax_w) + softmax_b
    #计算输出的softmax
    #output和上一步的logits具有相同的维度和数据类型，只是经过了tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
    output = tf.nn.softmax(logits)

    return output, logits

def model_loss(target, logits, num_class):
    '''
    计算交叉熵损失
    :param target:标签
    :param logits: 预测输出
    :param num_class: 字符的种类数目
    :return:
    '''
    #将标签生成为onehot向量
    y_one_hot = tf.one_hot(target, num_class)

    #计算损失
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
    loss = tf.reduce_mean(loss)

    return loss

def model_optimizer(learning_rate, loss, clip_val):
    '''
    【【LSTM不会有梯度消失的问题，但可能有梯度爆炸的问题，进行梯度裁剪，防止梯度爆炸】】
    :param learning_rate:学习率
    :param loss: 损失
    :param clip_val: 裁剪梯度，这里的值是5，也就是把所有的梯度剪裁到-5到5之间
    :return:
    '''
    #设置学习率
    #tran_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tran_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    #获取所有的训练变量
    allTvars = tf.trainable_variables()
    #loss对所有的训练变量进行求导，并进行梯度裁剪
    #tf.gradients(ys,xs)，xs，ys可以是变量或者list，表示所有ys对xs求导。返回值是xs大小的list。https://blog.csdn.net/hustqb/article/details/80260002
    all_grads, _ = tf.clip_by_global_norm(tf.gradients(loss, allTvars), clip_norm=clip_val)
    #将裁剪后的梯度重新应用于所有变量
    optimizer = tran_op.apply_gradients(zip(all_grads, allTvars))

    return optimizer

class char_RNN:
    def __init__(self, vocab, n_seqs = 10, n_sequencd_length = 30, lstm_num_units=128, keep_prob=0.5, num_layers=3,
                 learning_rate=0.01, clip_val=5):
        '''
        初始化模型
        :param vocab:字符集合
        :param n_seqs: 每次训练的样本数目，也就是batch_size
        :param n_sequencd_length: 每个样本的长度
        :param lstm_num_units: lstm节点中隐层节点数目
        :param keep_prob: 随机drop比例，防止过拟合
        :param num_layers: lstm节点层数
        :param learning_rate: 学习率
        :param clip_val: 梯度最大值，防止梯度爆炸，如果梯度过大，则依据此值进行裁剪
        '''

        #初始化模型的input和target
        self.input, self.target = model_input(n_seqs=n_seqs, n_sequencd_length=n_sequencd_length)
        #构建lstm
        cell, self.init_state = model_lstm(lstm_num_units=lstm_num_units, keep_prob=keep_prob, num_layers=num_layers,
                                      n_seqs=n_seqs)

        #对输入进行ｅｍｂｅｄｄｉｎｇ
        self.src_embedding = tf.get_variable(
            "src_emb", [len(vocab), lstm_num_units])
        src_emb = tf.nn.embedding_lookup(self.src_embedding, self.input)

        #【【dynamic_rnn指的是这里的LSTM单元可以接收多个时间步的数据，如果是静态RNN，那么时间步的数量是定死的】】
        #init_state是model_lstm中定义的LSTM单元的初始状态，为0
        #outputs的维度是[batch_size,max_time, HIDDEN_SIZE]
        outputs, self.state = tf.nn.dynamic_rnn(cell, src_emb, initial_state=self.init_state)

        #计算该batch的输出
        self.predtion, logits = model_output(lstm_output=outputs, in_size=lstm_num_units, out_size=len(vocab))

        #计算损失
        self.loss = model_loss(target=self.target, logits = logits, num_class=len(vocab))

        #梯度裁剪
        self.optimizer = model_optimizer(learning_rate=learning_rate, loss=self.loss, clip_val=clip_val)


if __name__ == '__main__':
    #加载数据
    vocab, vocab2Int, int2Vocab, xs,ys = generated_data()
    #初始化模型
    char_rnn = char_RNN(vocab=vocab, n_seqs = n_seqs, n_sequencd_length = n_sequencd_length,
                        lstm_num_units=lstm_num_units, keep_prob=keep_prob, num_layers=num_layers,
                 learning_rate=learning_rate, clip_val=5)

    saver = tf.train.Saver()

    #设置训练轮数
    epochs = 50
    #全局计数,一个batch就加一次count
    count = 0

    with tf.Session() as sess:
        #初始化所有变量
        sess.run(tf.global_variables_initializer())

        #找有没有已经训练的模型
        ckpt = tf.train.get_checkpoint_state(os.path.abspath('checkpoint'))
        if ckpt is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Using trained model:' + ckpt.model_checkpoint_path)

        #进行轮数迭代
        for epoch in range(epochs):
            #每次获取一个batch，进行训练
            for x, y in get_a_batch(xs,ys,n_seqs):
                count += 1

                feed = {
                    char_rnn.input:x,
                    char_rnn.target:y
                }

                #这三个节点都要执行，但是只有loss需要返回结果，所以另两个用_代替了
                _, loss, _ = sess.run([char_rnn.state, char_rnn.loss, char_rnn.optimizer], feed_dict=feed)

                #定期打印数据
                if count % 5 == 0:
                    print('-----------------------------')
                    print('轮数：%d:%d' % (epoch + 1, epochs))
                    print('训练步数：%d' % (count))
                    print('训练误差:%.4f' % (loss))
            #定期保存ckpt
            if epoch % 1 == 0:
                saver.save(sess, 'checkpoint/model.ckpt',global_step=count)

        saver.save(sess, 'checkpoint/model.ckpt', global_step=count)
