#include "relml/agent/Agent.h"

#include <iostream>
#include <string>

using namespace relml;

int main(int argc, char* argv[]) {
    std::string message = (argc > 1) ? argv[1] : "Hello! Say hi in one short sentence.";

    AgentConfig cfg;
    Agent agent(cfg);

    std::cout << "Sending: " << message << "\n\n";
    AgentResponse r = agent.send(message);
    std::cout << "Response: " << r.text << "\n";
    std::cout << "(in: " << r.input_tokens << " tok, out: " << r.output_tokens << " tok)\n";

    return 0;
}
