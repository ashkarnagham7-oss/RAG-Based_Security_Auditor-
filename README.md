# RAG-Based_Security_Auditor
A source-code vulnerability detection project that investigates whether retrieval-augmented generation can provide more grounded and reliable security analysis than relying only on an LLM’s pretrained knowledge and general reasoning capabilities.

The main objective of the project is to detect vulnerable and non-vulnerable functions by grounding the reasoning process in an external knowledge base containing labeled secure and vulnerable code examples. Instead of asking the LLM to make security decisions solely from what it learned during training, the system retrieves relevant vulnerability knowledge and supplies it together with the function under analysis. This makes the decision process explicitly dependent on external, inspectable evidence.

Two retrieval approaches are currently examined:

BGE-M3-based retrieval, where functional descriptions of source-code functions are embedded and matched against semantically similar vulnerability records.
UniXcoder-based retrieval, where source code itself is embedded and compared directly with vulnerable code examples in the knowledge base.

The retrieved records are then provided to an LLM-based reasoning stage, which classifies each function as vulnerable, not vulnerable, or uncertain, and can additionally identify the associated CWE and supporting evidence.

The two RAG configurations are evaluated against labeled ground truth and compared with Bandit, a static analysis tool, to study their differences in vulnerability detection performance, false positives, false negatives, and CWE identification.
