#pragma leco app
#include <stdio.h>

import jason;
import jute;
import print;
import traits;
import vee;
import yoyo;

using namespace traits::ints;
namespace j = jason::ast;
namespace jn = j::nodes;

// static constexpr const auto n_ctx = 1024;
// static constexpr const auto n_embed = 768;
// static constexpr const auto n_eps = 1e-05;
// static constexpr const auto n_head = 12;
static constexpr const auto n_layer = 12;
// static constexpr const auto n_vocab = 50257;

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
static auto create_buffer(int layer, const char * a, const char * b, auto mem) {
  char buf[1024] {};
  snprintf(buf, sizeof(buf), "h.%d.%s.%s", layer, a, b);
  return create_buffer(jute::view::unsafe(buf), mem);
}

int main() try {
  auto i = vee::create_instance("gptgpu");
  auto dbg = vee::create_debug_utils_messenger();
  const auto & [pd, qf] = vee::find_physical_device_with_universal_queue(nullptr);
  auto d = vee::create_single_queue_device(pd, qf);
  auto q = vee::get_queue_for_family(qf);

  auto mem = load_model(pd);
  auto json = jason::parse(*g_head);
  g_config = &j::cast<jn::dict>(json);

  auto wte = create_buffer("wte.weight", *mem); // n_vocab x n_embed
  auto wpe = create_buffer("wpe.weight", *mem); // n_ctx x n_embed
  auto lnfb = create_buffer("ln_f.bias", *mem); // n_embed x 1
  auto lnfw = create_buffer("ln_f.weight", *mem); // n_embed x 1

  struct {
    vee::buffer ln1_w {};
    vee::buffer ln1_b {};
    vee::buffer attn_w {};
    vee::buffer attn_b {};
    vee::buffer attn_pw {};
    vee::buffer attn_pb {};

    vee::buffer ln2_w {};
    vee::buffer ln2_b {};
    vee::buffer mlp_w {};
    vee::buffer mlp_b {};
    vee::buffer mlp_pw {};
    vee::buffer mlp_pb {};
  } layers[n_layer] {};
  for (auto i = 0; i < n_layer; i++) {
    layers[i].ln1_w = create_buffer(i, "ln_1", "weight", *mem);
    layers[i].ln1_b = create_buffer(i, "ln_1", "bias", *mem);
    layers[i].attn_w = create_buffer(i, "attn.c_attn", "weight", *mem);
    layers[i].attn_b = create_buffer(i, "attn.c_attn", "bias", *mem);
    layers[i].attn_pw = create_buffer(i, "attn.c_proj", "weight", *mem);
    layers[i].attn_pb = create_buffer(i, "attn.c_proj", "bias", *mem);

    layers[i].ln2_w = create_buffer(i, "ln_2", "weight", *mem);
    layers[i].ln2_b = create_buffer(i, "ln_2", "bias", *mem);
    layers[i].attn_w = create_buffer(i, "mlp.c_fc", "weight", *mem);
    layers[i].attn_b = create_buffer(i, "mlp.c_fc", "bias", *mem);
    layers[i].attn_pw = create_buffer(i, "mlp.c_proj", "weight", *mem);
    layers[i].attn_pb = create_buffer(i, "mlp.c_proj", "bias", *mem);
  }

} catch (...) {
  return 1;
}
