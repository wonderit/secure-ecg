# Data split
import torch
import syft as sy
from syft.frameworks.torch.crypto.securenn import (
    private_compare,
    decompose,
    share_convert,
    relu_deriv,
    msb
)

# Set everything up
hook = sy.TorchHook(torch)
L = 2 ** 62

alice = sy.VirtualWorker(id="alice", hook=hook)
bob = sy.VirtualWorker(id="bob", hook=hook)
james = sy.VirtualWorker(id="james", hook=hook)
# test_beta_sh = (
#     torch.LongTensor([-13, 2, 5])
#     .fix_precision()
#     .share(alice, bob, crypto_provider=james, field=L)
#     .child
# )
# t = test_beta_sh.get()
# t = t.float_precision()
# print(t)
x_bit_sh = decompose(torch.LongTensor([13])).share(alice, bob, crypto_provider=james, field=67).child
r = torch.LongTensor([12]).send(alice, bob).child

# beta = torch.LongTensor([1]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# print('betap', beta_p)
# assert not beta_p
#
# beta = torch.LongTensor([0]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# assert beta_p
#
# # Big values
# x_bit_sh = decompose(torch.LongTensor([2 ** 60])).share(alice, bob, crypto_provider=james).child
# r = torch.LongTensor([2 ** 61]).send(alice, bob).child
#
# beta = torch.LongTensor([1]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# assert beta_p
#
# beta = torch.LongTensor([0]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# assert not beta_p
#
# # Multidimensional tensors
# x_bit_sh = (
#     decompose(torch.LongTensor([[13, 44], [1, 28]]))
#     .share(alice, bob, crypto_provider=james)
#     .child
# )
# r = torch.LongTensor([[12, 44], [12, 33]]).send(alice, bob).child
#
# beta = torch.LongTensor([1]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# assert (beta_p == torch.tensor([[0, 1], [1, 1]])).all()
#
# beta = torch.LongTensor([0]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# assert (beta_p == torch.tensor([[1, 0], [0, 0]])).all()
#
# # Negative values
# x_val = -105 % 2 ** 62
# r_val = -52 % 2 ** 62  # The protocol works only for values in Zq
# x_bit_sh = decompose(torch.LongTensor([x_val])).share(alice, bob, crypto_provider=james).child
# r = torch.LongTensor([r_val]).send(alice, bob).child
#
# beta = torch.LongTensor([1]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# assert beta_p
#
# beta = torch.LongTensor([0]).send(alice, bob).child
# beta_p = private_compare(x_bit_sh, r, beta)
# assert not beta_p

"""
This is a light test as share_convert is not used for the moment
"""
# L = 2 ** 64
# x_bit_sh = (
#     torch.LongTensor([13, 3567, 2 ** 60])
#     .share(alice, bob, crypto_provider=james, field=L)
#     .child
# )
#
# res = share_convert(x_bit_sh)
# assert res.field == L - 1
# assert (res.get() % L == torch.LongTensor([13, 3567, 2 ** 60])).all()

x_sh = torch.tensor([10, -1, -3]).share(alice, bob, crypto_provider=james, field=L).child
r = relu_deriv(x_sh)

assert (r.get() == torch.tensor([1, 0, 0])).all()