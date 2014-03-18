#include <string>

#include "common.h"

class VideoMaker {
public:
   virtual void addFrame(Vector4* rgba_array, int fw, int fh) = 0;
   virtual void finish() = 0;
   virtual ~VideoMaker();
};

VideoMaker* VideoMaker_Create(const std::string& filename, int width, int height);
