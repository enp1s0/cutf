#ifndef __CUTF_NVTX_HPP__
#define __CUTF_NVTX_HPP__
#include <string>
#include <nvtx3/nvToolsExt.h>

namespace cutf {
namespace nvtx {
inline void range_push(
		const std::string name
		) {
	unsigned sid = 0;
	for (const auto c : name) {
		sid += static_cast<unsigned>(c);
	}

	const std::uint32_t color_list[] = {0x057dcdu, 0xebeef1u, 0xc197d2u, 0xd3b1c2u, 0x94c973, 0xe1c391};
	const auto cid = sid % (sizeof(color_list) / sizeof(std::uint32_t));

	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color_list[cid];
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name.c_str();
	nvtxRangePushEx(&eventAttrib);
}

inline void range_pop() {
	nvtxRangePop();
}
} // namespace nvtx
} // namespace cutf
#endif
