# (1)コンパイラ
CC  = g++
# (2)コンパイルオプション
CFLAGS    =

CXX=g++
INCDIR=-I/usr/include/hdf5/serial
LIBDIR=-L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib
CFLAGS=-Wall -O2 $(INCLUDEDIR) $(LIBDIR) -std=c++11 -fopenmp
LINK=-lhdf5_cpp -lhdf5

# (3)実行ファイル名
TARGET  = VeladjustCxx
# (4)コンパイル対象のソースコード
SRCS    = main.cpp
# (5)オブジェクトファイル名
OBJS    = $(SRCS:.cpp=.o)
 
# (8)追加するライブラリファイル
LIBS    = 

# (9)ターゲットファイル生成
$(TARGET): $(OBJS)
	$(CC) -std=c++11 -fopenmp -o $@ $^ $(LIBDIR) $(LIBS) $(LINK)

# (10)オブジェクトファイル生成
$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

# (11)"make all"で make cleanとmakeを同時に実施。
all: clean $(OBJS) $(TARGET)
# (12).oファイル、実行ファイル、.dファイルを削除
clean:
	-rm -f $(OBJS) $(TARGET) *.d
