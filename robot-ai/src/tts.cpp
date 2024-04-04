#include <opendaq/opendaq.h>
#include <opendaq/packet.h>
#include <stdlib.h>
#include <format>
#include <iostream>

auto main(int argc, char* argv[]) -> int
{
    auto instance = daq::Instance();
    auto device = instance.addDevice("daq.opcua://192.168.10.1");

    daq::SignalPtr voice_signal;
    for (auto signal : device.getSignalsRecursive())
    {
        if (signal.getName() == "robot_voice")
        {
            voice_signal = signal;
            break;
        }
    }

    if (!voice_signal.assigned())
        return 1;

    std::cout << "Found signal" << std::endl;

    auto reader = daq::PacketReader(voice_signal);

    while (true)
    {
        auto packet = reader.read();
        if (packet.assigned() && packet.getType() == daq::PacketType::Data)
        {
            daq::DataPacketPtr data_packet = packet;
            const char *str = static_cast<char*>(data_packet.getRawData());
            const auto cmd = std::format("echo \"{}\" | espeak -s 160 -p 50 -a 200 -g 4 -k 5", str);
            system(cmd.c_str());
        }
    }
    

    system("echo \"Hello\" | espeak");
    return 0;
}