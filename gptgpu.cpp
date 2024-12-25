#pragma leco app
#pragma leco add_shader "gptgpu.lnorm.comp"
#pragma leco add_shader "gptgpu.linear.comp"
#pragma leco add_shader "gptgpu.qkv.comp"
#pragma leco add_shader "gptgpu.smax.comp"

#include <stdio.h>

import gpt2;
import jason;
import jojo;
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

static auto create_local_buffer(unsigned len, auto mem, auto & acc) {
  auto buf = vee::create_buffer(len, vee::buffer_usage::storage_buffer);
  vee::bind_buffer_memory(*buf, mem, acc);
  acc += len;
  return buf;
}

static auto create_pipeline(jute::view shd, auto pl) {
  auto k_0 = vee::create_shader_module_from_resource(shd);
  return vee::create_compute_pipeline(pl, *k_0, "main");
}

static void print_token(const auto & vocab, int tk) {
  putf("%5d ", tk);
  if (tk == 198) {
    putln();
    return;
  }
  for (auto &[txt, id]: vocab) {
    auto id_i = j::cast<jn::number>(id).integer();
    if (id_i != tk) continue;
    putln(txt);
    return;
  }
  putln();
}

int main() try {
  auto vocab_file = jojo::read_cstr("out/vocab.json");
  auto vocab_json = jason::parse(vocab_file);
  auto & vocab = j::cast<jn::dict>(vocab_json);

  auto i = vee::create_instance("gptgpu");
  auto dbg = vee::create_debug_utils_messenger();
  const auto & [pd, qf] = vee::find_physical_device_with_universal_queue(nullptr);
  auto d = vee::create_single_queue_device(pd, qf);
  auto q = vee::get_queue_for_family(qf);

  auto mem = gpt2::load(pd);

  auto wte = gpt2::create_st_buffer("wte.weight"); // n_vocab x n_embed
  auto wpe = gpt2::create_st_buffer("wpe.weight"); // n_ctx x n_embed
  auto lnfb = gpt2::create_st_buffer("ln_f.bias"); // n_embed x 1
  auto lnfw = gpt2::create_st_buffer("ln_f.weight"); // n_embed x 1

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
  } l[n_layer] {};
  for (auto i = 0; i < n_layer; i++) {
    l[i].ln1_w = gpt2::create_st_buffer(i, "ln_1", "weight");
    l[i].ln1_b = gpt2::create_st_buffer(i, "ln_1", "bias");
    l[i].attn_w = gpt2::create_st_buffer(i, "attn.c_attn", "weight");
    l[i].attn_b = gpt2::create_st_buffer(i, "attn.c_attn", "bias");
    l[i].attn_pw = gpt2::create_st_buffer(i, "attn.c_proj", "weight");
    l[i].attn_pb = gpt2::create_st_buffer(i, "attn.c_proj", "bias");

    l[i].ln2_w = gpt2::create_st_buffer(i, "ln_2", "weight");
    l[i].ln2_b = gpt2::create_st_buffer(i, "ln_2", "bias");
    l[i].attn_w = gpt2::create_st_buffer(i, "mlp.c_fc", "weight");
    l[i].attn_b = gpt2::create_st_buffer(i, "mlp.c_fc", "bias");
    l[i].attn_pw = gpt2::create_st_buffer(i, "mlp.c_proj", "weight");
    l[i].attn_pb = gpt2::create_st_buffer(i, "mlp.c_proj", "bias");
  }

  // TODO: implement the encoder properly
  auto tk_buf = vee::create_buffer(n_ctx * sizeof(unsigned), vee::buffer_usage::storage_buffer);
  auto tk_mem = vee::create_host_buffer_memory(pd, *tk_buf);
  vee::bind_buffer_memory(*tk_buf, *tk_mem, 0);

  auto in_tks = static_cast<unsigned *>(vee::map_memory(*tk_mem));
  unsigned tks = 0;
  in_tks[tks++] = j::cast<jn::number>(vocab["Who"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["Ġis"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["ĠAlfred"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["ĠNobel"]).integer();
  in_tks[tks++] = j::cast<jn::number>(vocab["?"]).integer();
  for (auto i = 0; i < tks; i++) print_token(vocab, in_tks[i]);
  vee::unmap_memory(*tk_mem);

  gpt2::stages::wtewpe wtewpe { pd, *wte, *wpe, *tk_buf };

  static constexpr const auto max_sets = 16;
  auto dpool = vee::create_descriptor_pool(max_sets, { vee::storage_buffer(max_sets * 4) });

  auto cpool = vee::create_command_pool(qf);
  auto cb = vee::allocate_primary_command_buffer(*cpool);

  vee::begin_cmd_buf_one_time_submit(cb);
  wtewpe.cmd_dispatch(cb, tks);
  vee::end_cmd_buf(cb);

  auto f = vee::create_fence_signaled();
  vee::wait_and_reset_fence(*f);
  vee::queue_submit({
    .queue = q,
    .fence = *f,
    .command_buffer = cb,
  });
  vee::wait_for_fence(*f);
} catch (...) {
  return 1;
}
