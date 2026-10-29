#ifndef BITWISE_HPP
#define BITWISE_HPP

#define BIT_SET(bitstr, i) (bitstr |= 1<<i)
#define BIT_GET(bitstr, i) ((bitstr>>i) & 1)

#define ONES_MASK(n) ((1 << (n)) - 1)


class BoolArray16
{
public:
    __device__ inline void set(const uint8_t i) { BIT_SET(barray, i);  }
    __device__ inline bool is (const uint8_t i) { return BIT_GET(barray, i); }

private:
    uint16_t barray = 0;
};

#endif