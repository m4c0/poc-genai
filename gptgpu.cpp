#pragma leco app

import jason;
import jute;
import print;
import traits;
import vee;
import yoyo;

using namespace traits::ints;
namespace j = jason::ast;
namespace jn = j::nodes;

static jute::heap g_cnt {};
static const jn::dict * g_config {};

static auto load_model(auto pd) {
  auto f = yoyo::file_reader::open("out/model.safetensors");
  auto len = f.fmap(yoyo::size()).take([](auto msg) { die("error reading model: ", msg); });

  auto res = vee::create_host_buffer_memory(pd, len);
  auto ptr = static_cast<float *>(vee::map_memory(*res));
  f.fmap(yoyo::read(ptr, len)).take([](auto msg) { die("error reading model: ", msg); });

  auto sz = *reinterpret_cast<const uint64_t *>(ptr);
  g_cnt = jute::view { reinterpret_cast<const char *>(ptr) + 8, sz };

  vee::unmap_memory(*res);

  auto json = jason::parse(*g_cnt);
  g_config = &j::cast<jn::dict>(json);
  return res;
}

int main() try {
  auto i = vee::create_instance("gptgpu");
  auto dbg = vee::create_debug_utils_messenger();
  const auto & [pd, qf] = vee::find_physical_device_with_universal_queue(nullptr);
  auto d = vee::create_single_queue_device(pd, qf);
  auto q = vee::get_queue_for_family(qf);

  auto model_mem = load_model(pd);
} catch (...) {
  return 1;
}
