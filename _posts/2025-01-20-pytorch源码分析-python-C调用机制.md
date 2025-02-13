---
title: pytorch源码分析-python-C调用机制
categories:
  - Compiller
  - DeepLearning
tags:
  - Transformer
  - Pytorch
share: true
---

在前文[[./2024-08-12-TransformerModel-基于minGPT理解|2024-08-12-TransformerModel-基于minGPT理解]]中分析基于pytorch实现的简单的transform模型。本文关注基于pytorch做推理时，pytorch源码是如何实现的。

## 模型代码
前文训练好的nano-gpt模型可以通过如下代码加载运行：
```python
import os
import torch
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = 3
model_config.block_size = 11
model = GPT(model_config)

ckpt_path = os.path.join("./out", "demo_model.pt")
model.load_state_dict(torch.load(ckpt_path))


inp = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to('cpu')
cat = model.generate(inp, 6, do_sample=False)
print(cat)
```

其中核心是调用GPT模型的`generate()`接口进行推理：
```python
@torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```
推理过程在`self(idx_cond)`语句中触发前向传播，此处的`Self`是继承自`torch.nn.Model`的`class GPT`，该语句会调用`torch.nn.Model`的`__call__`函数执行调用。

## pytorch源码 - forward调用
在pytorch源码中对`__call__`方法做了包装：
```python
__call__: Callable[..., Any] = _wrapped_call_impl

def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:
        return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
    else:
        return self._call_impl(*args, **kwargs)

# torchrec tests the code consistency with the following code
# fmt: off
def _call_impl(self, *args, **kwargs):
    forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
    # If we don't have any hooks, we want to skip the rest of the logic in
    # this function, and just call forward.
    if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
            or _global_backward_pre_hooks or _global_backward_hooks
            or _global_forward_hooks or _global_forward_pre_hooks):
        return forward_call(*args, **kwargs)

```
此处call提供了compile路径和直接调用路径`_compiled_call_impl`和`_call_impl`, 在`_call_impl`调用中，pytorch调用了`torch.nn.Model`的`forward()`方法，该方法由模型定义，在本文中就是`GPT`类的`forward`方法。
```python
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

```

在`forward`方法中包含了模型的众多算子，transformer模型中的主要算子可以参考[前文](obsidian://open?vault=HunterNoteBook&file=Compiler%2FDLCompiler%2F2024-08-12-TransformerModel-%E5%9F%BA%E4%BA%8EminGPT%E7%90%86%E8%A7%A3)Transformer模型介绍。

## pytorch源码 - embedding层

在Transformer模型中，输入token首先需要经过embedding层进行编码，此处以embedding算子为例，分析pytorch的源码实现。

pytorch中算子通过c++实现，而API接口通过python提供，这样兼顾c++高性能和python高效率的优点。在python中使用c/c++函数扩展可以参考[python官方文档](https://docs.python.org/3/extending/extending.html)。

上述Transformer模型在词嵌入过程中`tok_emb = self.transformer.wte(idx)`，`wte`本身是一个`nn.Embedding`层：
```python
# model.py/GPT/__init__()
self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
```

nn.Embedding在pytorch源码`torch/nn/modules/sparse.py`中，在推理阶段，`forward`方法如下：
```python
def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
# F.embedding - torch/nn/functional.py
def embedding():
	#...
	return torch.embedding();
```

此处最终调用的`torch.embedding()`算子为c++实现：
```cpp
static PyMethodDef torch_functions_shard[] = {
	// ...
	{"embedding", castPyCFunctionWithKeywords(THPVariable_embedding), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
	// ...
};

// embedding
static PyObject * THPVariable_embedding(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
  
  auto dispatch_embedding = [](const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  };
  return wrap(dispatch_embedding(_r.tensor(0), _r.tensor(1), _r.toSymInt(2), _r.toBool(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

```

最终embedding层的计算逻辑调用到了c++实现的`at::embedding_symint()`函数中。

至此，本文从模型源码出发，分析了算子如何在pytorch框架中调用。pytorch利用python中c/c++扩展能力，将算子封装为python接口，实现性能和易用性的平衡。
