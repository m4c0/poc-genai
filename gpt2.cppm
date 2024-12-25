#pragma leco add_impl gpt2_metadata
export module gpt2;
export import :consts;
export import :smax;
export import :wtewpe;
import print;

namespace gpt2 {
  export vee::device_memory load(vee::physical_device pd);
  export vee::buffer create_st_buffer(jute::view key);
  export vee::buffer create_st_buffer(int layer, const char * a, const char * b);

  export void debug(vee::device_memory::type mem, unsigned r, unsigned c) {
    auto out = static_cast<float *>(vee::map_memory(mem));
    for (auto h = 0; h < r; h++) {
      if (h == 3) putln("...");
      if (h >= 3 && h < r - 3) continue;

      for (auto i = 0; i < c; i++) {
        if (i == 3) put("... ");
        if (i >= 3 && i < c - 3) continue;

        putf("%+1.3f ", out[h * c + i]);
      }
      putln();
    }
    vee::unmap_memory(mem);
  }
}
