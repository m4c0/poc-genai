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

static jute::heap g_head {};
static const jn::dict * g_config {};

static auto load_model(auto pd) {
  auto f = yoyo::file_reader::open("out/model.safetensors");
  auto len = f.fmap(yoyo::size()).take([](auto msg) { die("error reading model: ", msg); });

  auto res = vee::create_host_buffer_memory(pd, len);
  auto ptr = static_cast<float *>(vee::map_memory(*res));
  f.fmap(yoyo::read(ptr, len)).take([](auto msg) { die("error reading model: ", msg); });

  auto sz = *reinterpret_cast<const uint64_t *>(ptr);
  g_head = jute::view { reinterpret_cast<const char *>(ptr) + 8, sz };

  vee::unmap_memory(*res);
  return res;
}
static auto create_buffer(jute::view key, auto mem) {
  auto & root = *g_config;
  auto & v = j::cast<jn::dict>(root[key]);
  auto dtype = j::cast<jn::string>(v["dtype"]).str();
  if (*dtype != "F32") die("unsupported dtype ", *dtype);

  auto & offs = j::cast<jn::array>(v["data_offsets"]);
  auto start = j::cast<jn::number>(offs[0]).integer();
  auto end = j::cast<jn::number>(offs[1]).integer();
  //if (end < start || end - start > g_cnt.size()) die("invalid offsets ", start, "~", end);

  unsigned len = end - start;
  auto buf = vee::create_buffer(len, vee::buffer_usage::storage_buffer);
  vee::bind_buffer_memory(*buf, mem, start - g_head.size() - 8);
  return buf;
}

int main() try {
  auto i = vee::create_instance("gptgpu");
  auto dbg = vee::create_debug_utils_messenger();
  const auto & [pd, qf] = vee::find_physical_device_with_universal_queue(nullptr);
  auto d = vee::create_single_queue_device(pd, qf);
  auto q = vee::get_queue_for_family(qf);

  auto model_mem = load_model(pd);
  auto json = jason::parse(*g_head);
  g_config = &j::cast<jn::dict>(json);

  auto wte = create_buffer("wte.weight", *model_mem); // n_vocab x n_embed
  auto wpe = create_buffer("wpe.weight", *model_mem); // n_ctx x n_embed
  auto lnfb = create_buffer("ln_f.bias", *model_mem); // n_embed x 1
  auto lnfw = create_buffer("ln_f.weight", *model_mem); // n_embed x 1
} catch (...) {
  return 1;
}
