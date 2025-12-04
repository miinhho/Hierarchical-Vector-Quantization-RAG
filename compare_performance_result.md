Fetching 20 Newsgroups dataset (this may take a moment)...
Dataset size: 5000 real documents from 20 Newsgroups

=== Running Baseline: Flat RAG ===
Flat RAG Ingestion Time: 2607.7559s
Flat RAG Index Size (Approx RAM): 50592.00 KB
Flat RAG result: [{'id': '1c68e90b-6e82-4731-b705-2a18f74456e9', 'text': 'cated at the Lyndon B. Johnson Space Center in\n    Houston, Texas, and will involved a 1-year training and evaluation\n    program.\n\n    Space Shuttle Program Description\n    ---------------------------------\n\n    The numerous successful flights of the Space Shuttle have demonstrated\n    that operation and experimental investigations in space are becoming\n    routine. The Space Shuttle Orbiter is launched into, and maneuvers in\n    the Earth orbit performing missions lastling up to 30 days. It th', 'score': 0.5576266050338745}, {'id': 'dbfdd99c-bf61-46ec-bffe-f98a539bed9c', 'text': "e\n    *Science*, V. 257, p. 1487-1489 (11 September 1992). For gory technical\n    detail, see the many articles in the same issue.\n\n\n    OTHER SPACE SCIENCE MISSIONS (note: this is based on a posting by Ron\n    Baalke in 11/89, with ISAS/NASDA information contributed by Yoshiro\n    Yamada (yamada@yscvax.ysc.go.jp). I'm attempting to track changes based\n    on updated shuttle manifests; corrections and updates are welcome.\n\n    1993 Missions\n\to ALEXIS [spring, Pegasus]\n\t    ALEXIS (Array of Low-E", 'score': 0.8464583158493042}, {'id': '50e3b762-c849-4346-ac2f-a3501b1a98cb', 'text': 'Archive-name: space/schedule\nLast-modified: $Date: 93/04/01 14:39:23 $\n\nSPACE SHUTTLE ANSWERS, LAUNCH SCHEDULES, TV COVERAGE\n\n    SHUTTLE LAUNCHINGS AND LANDINGS; SCHEDULES AND HOW TO SEE THEM\n\n    Shuttle operations are discussed in the Usenet group sci.space.shuttle,\n    and Ken Hollis (gandalf@pro-electric.cts.com) posts a compressed version\n    of the shuttle manifest (launch dates and other information)\n    periodically there. The manifest is also available from the Ames SPACE\n    archive i', 'score': 0.8540892004966736}, {'id': '6d4bcba7-90b5-40bb-8c39-4a12ebdd960d', 'text': 'n specialists are required to\n    have a detailed knowledge of Shuttle systems, as well as detailed\n    knowledge of the operational characteristics, mission requirements and\n    objectives, and supporting systems and equipment for each of the\n    experiments to be conducted on their assigned missions. Mission\n    specialists will perform extra-vehicular activities, payload handling\n    using the remote manipulator system, and perform or assist in specific\n    experimental operations.\n\n    Astro', 'score': 0.8571603298187256}, {'id': '64b7e849-7297-4aa2-865f-82f1501b31d8', 'text': 'n the deployment and retrieval of satellites utilizing\n    the remote manipulator system, in extra-vehicular activities, and other\n    payload operations.\n\n    Mission Specialist Astronaut\n\n    Mission specialist astronauts, working with the commander and pilot,\n    have overall responsibility for the coordination of Shuttle operations\n    in the areas of crew activity planning, consumables usage, and\n    experiment and payload operations. Mission specialists are required to\n    have a detailed ', 'score': 0.8814061880111694}]
Flat RAG Query Time: 0.0695s

=== Running Proposed: HiRAG ===
Encoding 16864 chunks...
DEBUG: Level 0 Precision: float32, Settings: {0: 'float32', 1: 'float16', 2: 'int8'}
Saving Level 0...
Building level 1 from 16864 nodes...
Building level 2 from 3372 nodes...
Ingestion complete.
HiRAG Ingestion Time: 1328.6479s
HiRAG Vector Index Size (Disk): 56161.70 KB
DEBUG: Top layer is 2. Store ntotal: 674
DEBUG: Top layer search returned 10 results
DEBUG: Drilling down to layer 1. Candidates: 10
DEBUG: Found 57 children
DEBUG: Retrieved 57 child nodes
DEBUG: Scored 57 children
DEBUG: Selected 5 candidates for next layer
DEBUG: Drilling down to layer 0. Candidates: 5
DEBUG: Found 26 children
DEBUG: Retrieved 26 child nodes
DEBUG: Scored 26 children
DEBUG: Selected 5 candidates for next layer
HiRAG result: [{'id': 'c0b28b6c-3317-4c3e-b056-7b033ff86210', 'text': 'cated at the Lyndon B. Johnson Space Center in\n    Houston, Texas, and will involved a 1-year training and evaluation\n    program.\n\n    Space Shuttle Program Description\n    ---------------------------------\n\n    The numerous successful flights of the Space Shuttle have demonstrated\n    that operation and experimental investigations in space are becoming\n    routine. The Space Shuttle Orbiter is launched into, and maneuvers in\n    the Earth orbit performing missions lastling up to 30 days. It th', 'metadata': {'source_idx': 12816}}, {'id': '40cacb9d-61b2-4b1f-81ed-b5e01d563dc3', 'text': 'Archive-name: space/schedule\nLast-modified: $Date: 93/04/01 14:39:23 $\n\nSPACE SHUTTLE ANSWERS, LAUNCH SCHEDULES, TV COVERAGE\n\n    SHUTTLE LAUNCHINGS AND LANDINGS; SCHEDULES AND HOW TO SEE THEM\n\n    Shuttle operations are discussed in the Usenet group sci.space.shuttle,\n    and Ken Hollis (gandalf@pro-electric.cts.com) posts a compressed version\n    of the shuttle manifest (launch dates and other information)\n    periodically there. The manifest is also available from the Ames SPACE\n    archive i', 'metadata': {'source_idx': 13927}}, {'id': 'fb6e1f6b-24ac-4239-9eb6-cf589d3dd199', 'text': 'n specialists are required to\n    have a detailed knowledge of Shuttle systems, as well as detailed\n    knowledge of the operational characteristics, mission requirements and\n    objectives, and supporting systems and equipment for each of the\n    experiments to be conducted on their assigned missions. Mission\n    specialists will perform emander and pilot,\n    have overall responsibility for the coordination of Shuttle operations\n    in the areas of crew activity planning, consumables usage, and\n    experiment and payload operations. Mission specialists are required to\n    have a detailed ', 'metadata': {'source_idx': 12820}}, {'id': '2828df73-a96e-4b35-a9a0-7a8352923ca0', 'text': 'ogram\n    ---------------------------\n\n    The National Aeronautics and Space Administration (NASA) has a need for\n    Pilot Astronaut Candidates and Mission Specialist Astronaut Candidates\n    to support the Space Shuttle Program. NASA is now accepting on a\n    continuous basis and plans to select astronaut candidates as needed.\n\n    Persons from both the civilian sector and the military services will be\n    considered.\n\n    All positions are located at the Lyndon B. Johnson Space Center in\n   ', 'metadata': {'source_idx': 12815}}] 
HiRAG Query Time: 0.0027s

=== Comparison Summary ===
Metric               | Flat RAG        | HiRAG           | Improvement
----------------------------------------------------------------------
Ingestion Time       | 2607.7559s       | 1328.6479s       | 1.96x (Slower is expected)
Index Size           | 50592.00 KB       | 56161.70 KB       | 0.90x Smaller
Query Time           | 0.0695s       | 0.0027s       | 25.80x Faster