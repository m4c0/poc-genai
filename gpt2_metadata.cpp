module;
#include <stdio.h>
module gpt2;
import hai;
import jason;
import jute;
import vee;
import yoyo;

namespace j = jason::ast;
namespace jn = j::nodes;

static hai::array<char> g_metadata_buffer {};
static jason::ast::node_ptr g_metadata_json {};
static vee::device_memory::type g_metadata_mem {};

static auto fail(jute::view msg) {
  die("error reading model: ", msg);
}

vee::device_memory gpt2::load(vee::physical_device pd) {
  auto f = yoyo::file_reader::open("out/model.safetensors");
  auto flen = f.fmap(yoyo::size()).take(fail);

  auto len = f.fmap([](auto & r) { return r.read_u64(); }).take(fail);
  g_metadata_buffer.set_capacity(len);

  f.fmap(yoyo::read(g_metadata_buffer.begin(), len)).take(fail);
  g_metadata_json = jason::parse({ g_metadata_buffer.begin(), g_metadata_buffer.size() });

  auto res = vee::create_host_memory(pd, flen);
  auto ptr = static_cast<char *>(vee::map_memory(*res));
  f.fmap(yoyo::read(ptr, flen - len - 8)).take(fail);
  vee::unmap_memory(*res);
  g_metadata_mem = *res;
  return res;
}

static auto parr_view(jute::view key) {
  auto & root = j::cast<jn::dict>(g_metadata_json);
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
vee::buffer gpt2::create_st_buffer(jute::view key) {
  auto [ start, end ] = parr_view(key);

  unsigned len = end - start;
  auto buf = vee::create_buffer(len, vee::buffer_usage::storage_buffer);
  vee::bind_buffer_memory(*buf, g_metadata_mem, start);
  return buf;
}
vee::buffer gpt2::create_st_buffer(int layer, const char * a, const char * b) {
  char buf[1024] {};
  snprintf(buf, sizeof(buf), "h.%d.%s.%s", layer, a, b);
  return create_st_buffer(jute::view::unsafe(buf));
}
