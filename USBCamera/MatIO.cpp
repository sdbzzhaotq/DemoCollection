#include "MatIO.h"
#include <fstream>

using namespace std;
using namespace cv;

namespace Utils {
    /**
     * 保存mat头的关键信息
     */
    typedef struct {
        int rows;//行数
        int cols;//列数
        int type;//类型
    } MatHeader;
}

bool Utils::write(const std::string &fileName, const cv::Mat &src) {
    Mat _src;
    // 保证mat是连续的
    if (!src.isContinuous()){
         _src = src.clone();
    } else{
        _src = src;
    }

    MatHeader matHeader{_src.rows, _src.cols, _src.type()};
    //printf("%d %d %d\n",matHeader.rows,matHeader.cols,matHeader.type);

    //打开文件
    ofstream out(fileName, ios::binary);

    //写入文件类型，表示为这个类型的数据，长度为两字节
    char fileType[3] = "mb";
    out.write(fileType, 2*sizeof(char));

    //写入Mat头
    out.write((char *) &matHeader, sizeof(MatHeader));

    //写入数据
    //http://blog.csdn.net/dcrmg/article/details/52294259
    //https://www.cnblogs.com/wangguchangqing/p/4016179.html
    out.write((char *) _src.data, _src.rows * _src.step[0]);

    out.flush();
    out.close();

    return true;
}

cv::Mat Utils::read(const std::string &fileName) {
    //打开文件
    ifstream in(fileName, ios::binary);

    char fileType[3]={'\0','\0','\0'};//初始化一个默认值

    //读取前两字节
    in.read(fileType, 2*sizeof(char));

    //读取Mat头
    MatHeader matHeader{0, 0, 0};
    in.read((char *) &matHeader, sizeof(MatHeader));
    //printf("%d %d %d\n",matHeader.rows,matHeader.cols,matHeader.type);

    //初始化一个Mat
    Mat mat(matHeader.rows, matHeader.cols, matHeader.type);

    //写入数据
    in.read((char *) mat.data, mat.rows * mat.step[0]);

    in.close();
    return mat;
}
