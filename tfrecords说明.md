tensorflow的example解析

example协议

在TensorFlow官方github文档里面，有个example.proto的文件,这个文件详细说明了TensorFlow里面的example协议，下面我将简要叙述一下。

tensorflow的example包含的是基于key-value对的存储方法，其中key是一个字符串，其映射到的是feature信息，feature包含三种类型：

BytesList：字符串列表  
FloatList：浮点数列表  
Int64List：64位整数列表  
以上三种类型都是列表类型，意味着都能够进行拓展,但是也是因为这种弹性格式，所以在解析的时候，需要制定解析参数，这个稍后会讲。  

在TensorFlow中，example是按照行读的，这个需要时刻记住，比如存储M×N矩阵，使用ByteList存储的话，需要M×N大小的列表，按照每一行的读取方式存放。  

tf.tain.example  

官方给了一个example的例子：  

An Example for a movie recommendation application:
   features {
     feature {
       key: "age"
       value { float_list {
         value: 29.0
       }}
     }
     feature {
       key: "movie"
       value { bytes_list {
         value: "The Shawshank Redemption"
         value: "Fight Club"
       }}
     }
     feature {
       key: "movie_ratings"
       value { float_list {
         value: 9.0
         value: 9.7
       }}
     }
     feature {
       key: "suggestion"
       value { bytes_list {
         value: "Inception"
       }}
     }

上面的例子中包含一个features，features里面包含一些feature，和之前说的一样，每个feature都是由键值对组成的，其key是一个字符串，其value是上面提到的三种类型之一。 

Example中有几个一致性规则需要注意： 

如果一个example的feature K 的数据类型是 T，那么所有其他的所有feature K都应该是这个数据类型
feature K 的value list的item个数可能在不同的example中是不一样多的，这个取决于你的需求
如果在一个example中没有feature k，那么如果在解析的时候指定一个默认值的话，那么将会返回一个默认值
如果一个feature k 不包含任何的value值，那么将会返回一个空的tensor而不是默认值
