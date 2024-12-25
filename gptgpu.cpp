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

static constexpr const auto n_ctx = 1024;
static constexpr const auto n_embed = 768;
// static constexpr const auto n_eps = 1e-05;
static constexpr const auto n_head = 12;
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
static auto parr_view(jute::view key) {
  auto & root = *g_config;
  auto & v = j::cast<jn::dict>(root[key]);
  auto dtype = j::cast<jn::string>(v["dtype"]).str();
  if (*dtype != "F32") die("unsupported dtype ", *dtype);

  auto & offs = j::cast<jn::array>(v["data_offsets"]);

  struct pair { int start, end; } res;
  res.start = j::cast<jn::number>(offs[0]).integer();
  res.end = j::cast<jn::number>(offs[1]).integer();
  //if (end < start || end - start > g_cnt.size()) die("invalid offsets ", start, "~", end);
  return res;
}
static auto create_buffer(jute::view key, auto mem) {
  auto [ start, end ] = parr_view(key);

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

static auto create_local_buffer(unsigned len, auto mem, auto & acc) {
  auto buf = vee::create_buffer(len, vee::buffer_usage::storage_buffer);
  vee::bind_buffer_memory(*buf, mem, acc);
  acc += len;
  return buf;
}
static auto create_local_buffer(jute::view sz, auto mem, auto & acc) {
  auto [ start, end ] = parr_view(sz);
  unsigned len = end - start;
  return create_local_buffer(len, mem, acc);
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

  auto mem = load_model(pd);
  auto json = jason::parse(*g_head);
  g_config = &j::cast<jn::dict>(json);

  auto wte = create_buffer("wte.weight", *mem); // n_vocab x n_embed
  auto wpe = create_buffer("wpe.weight", *mem); // n_ctx x n_embed
  auto lnfb = create_buffer("ln_f.bias", *mem); // n_embed x 1
  auto lnfw = create_buffer("ln_f.weight", *mem); // n_embed x 1

  auto l_mem = vee::create_local_buffer_memory(pd, 1);
  unsigned l_ptr {};

  // b1 = norm(b0)
  auto l_buf1 = create_local_buffer("wpe.weight", *l_mem, l_ptr);
  // b2 = qkv = linear(b1)
  auto l_buf2 = create_local_buffer("h.0.attn.c_attn.weight", *l_mem, l_ptr);
  // b3 = q @ k.T / sqrt(ctx) + mask
  auto l_buf3 = create_local_buffer(n_ctx * n_embed * 3 * n_head, *l_mem, l_ptr);
  // b4 = softmax(b3)
  auto l_buf4 = create_local_buffer(n_ctx * n_ctx * n_head, *l_mem, l_ptr);

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
    l[i].ln1_w = create_buffer(i, "ln_1", "weight", *mem);
    l[i].ln1_b = create_buffer(i, "ln_1", "bias", *mem);
    l[i].attn_w = create_buffer(i, "attn.c_attn", "weight", *mem);
    l[i].attn_b = create_buffer(i, "attn.c_attn", "bias", *mem);
    l[i].attn_pw = create_buffer(i, "attn.c_proj", "weight", *mem);
    l[i].attn_pb = create_buffer(i, "attn.c_proj", "bias", *mem);

    l[i].ln2_w = create_buffer(i, "ln_2", "weight", *mem);
    l[i].ln2_b = create_buffer(i, "ln_2", "bias", *mem);
    l[i].attn_w = create_buffer(i, "mlp.c_fc", "weight", *mem);
    l[i].attn_b = create_buffer(i, "mlp.c_fc", "bias", *mem);
    l[i].attn_pw = create_buffer(i, "mlp.c_proj", "weight", *mem);
    l[i].attn_pb = create_buffer(i, "mlp.c_proj", "bias", *mem);
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

  auto dsl_m2 = vee::create_descriptor_set_layout({
    vee::dsl_compute_storage(),
    vee::dsl_compute_storage(),
  });

  auto dsl_m4 = vee::create_descriptor_set_layout({
    vee::dsl_compute_storage(),
    vee::dsl_compute_storage(),
    vee::dsl_compute_storage(),
    vee::dsl_compute_storage(),
  });

  auto pl_m2 = vee::create_pipeline_layout({ *dsl_m2 });

  auto pl_m4 = vee::create_pipeline_layout({ *dsl_m4 });
  auto pl_m4k = vee::create_pipeline_layout(
    { *dsl_m4 },
    { vee::compute_push_constant_range<unsigned>() });

  auto ds_1 = vee::allocate_descriptor_set(*dpool, *dsl_m4);
  auto p_1 = create_pipeline("gptgpu.lnorm.comp.spv", *pl_m4);
  vee::update_descriptor_set_with_storage(ds_1, 0, wtewpe.buffer());
  vee::update_descriptor_set_with_storage(ds_1, 1, *l[0].ln1_w);
  vee::update_descriptor_set_with_storage(ds_1, 2, *l[0].ln1_b);
  vee::update_descriptor_set_with_storage(ds_1, 3, *l_buf1);

  auto ds_2 = vee::allocate_descriptor_set(*dpool, *dsl_m4);
  auto p_2 = create_pipeline("gptgpu.linear.comp.spv", *pl_m4k);
  vee::update_descriptor_set_with_storage(ds_2, 0, *l_buf1);
  vee::update_descriptor_set_with_storage(ds_2, 1, *l[0].attn_w);
  vee::update_descriptor_set_with_storage(ds_2, 2, *l[0].attn_b);
  vee::update_descriptor_set_with_storage(ds_2, 3, *l_buf2);

  auto ds_3 = vee::allocate_descriptor_set(*dpool, *dsl_m2);
  auto p_3 = create_pipeline("gptgpu.qkv.comp.spv", *pl_m2);
  vee::update_descriptor_set_with_storage(ds_3, 0, *l_buf2);
  vee::update_descriptor_set_with_storage(ds_3, 1, *l_buf3);

  auto ds_4 = vee::allocate_descriptor_set(*dpool, *dsl_m2);
  auto p_4 = create_pipeline("gptgpu.smax.comp.spv", *pl_m2);
  vee::update_descriptor_set_with_storage(ds_4, 0, *l_buf3);
  vee::update_descriptor_set_with_storage(ds_4, 1, *l_buf4);

  auto cpool = vee::create_command_pool(qf);
  auto cb = vee::allocate_primary_command_buffer(*cpool);

  vee::begin_cmd_buf_one_time_submit(cb);
  wtewpe.cmd_dispatch(cb);

  // mha
  // layer norm
  vee::cmd_bind_c_pipeline(cb, *p_1);
  vee::cmd_bind_c_descriptor_set(cb, *pl_m4, 0, ds_1);
  vee::cmd_dispatch(cb, n_ctx, 1, 1);
  vee::cmd_pipeline_barrier(cb, *l_buf1, vee::from_compute_to_compute);
  // linear attn
  vee::cmd_bind_c_pipeline(cb, *p_2);
  vee::cmd_bind_c_descriptor_set(cb, *pl_m4, 0, ds_2);
  vee::cmd_push_compute_constants(cb, *pl_m4k, &n_embed);
  vee::cmd_dispatch(cb, n_ctx, n_embed * 3, 1);
  vee::cmd_pipeline_barrier(cb, *l_buf2, vee::from_compute_to_compute);
  // qkv
  vee::cmd_bind_c_pipeline(cb, *p_3);
  vee::cmd_bind_c_descriptor_set(cb, *pl_m2, 0, ds_3);
  vee::cmd_dispatch(cb, n_head, n_ctx, n_embed * 3);
  vee::cmd_pipeline_barrier(cb, *l_buf3, vee::from_compute_to_compute);
  // softmax
  vee::cmd_bind_c_pipeline(cb, *p_4);
  vee::cmd_bind_c_descriptor_set(cb, *pl_m2, 0, ds_4);
  vee::cmd_dispatch(cb, n_head, n_ctx, n_ctx);
  vee::cmd_pipeline_barrier(cb, *l_buf4, vee::from_compute_to_compute);

  // ffn

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
