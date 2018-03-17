# 版本管理

| Author | Ver. | Date | Log |
|---|---|---|---|
| 成凯华 | 1.0 | 2018.03.16 | 目标识别PC端测试程序release |
|  |  |  |  |
|  |  |  |  |

# 文件树
* test.jpg
* test.xml
* test_predict_xml.exe
* graph_class6_nasnet.pb
* labels_list.txt

# 程序调用示例

test_predict_xml.exe test.jpg test.xml result.xml

	## 参数说明 
	1. img_name: 测试的图片
	2. xml_in_name: 测试图片对应的标记文件, 其中, 所有目标的名字已替换成“Unknown”
	3. xml_out_name: 生成的标记文件, 和输入的标记文件一样, 只是把目标的名字替换成识别出的类别
	4. pb_file_name: pb模型文件的全路径名, 可以空缺, 默认是当前路径下的"graph_class6_nasnet.pb"文件
	5. label_file_name: 记录类别名称的文件的全路径名, 可以空缺, 默认是当前路径下的"labels_list.txt"文件




