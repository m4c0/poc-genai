export module gpt2:consts;

export namespace gpt2 {
  constexpr const auto n_ctx = 1024;
  constexpr const auto n_embed = 768;
  constexpr const auto n_eps = 1e-05;
  constexpr const auto n_head = 12;
  constexpr const auto n_layer = 12;
  constexpr const auto n_vocab = 50257;
}
