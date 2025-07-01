"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

# Legacy prompts - kept for backward compatibility and comparison
# These have been replaced by the more efficient KG_EXTRACTION approach

PROMPTS["LEGACY_hi_entity_extraction"] = """
[Legacy Prompt - Use KG_EXTRACTION instead for better efficiency]
Given a text document that is potentially relevant to a list of entity types, identify all entities of those types.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}], normal_entity means that doesn't belong to any other types.
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. Return output in English as a single list of all the entities identified in step 1. Use **{record_delimiter}** as the list delimiter.

3. When finished, output {completion_delimiter}

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["LEGACY_hi_relation_extraction"] = """
[Legacy Prompt - Use KG_EXTRACTION instead for better efficiency]
Given a text document that is potentially relevant to a list of entities, identify all relationships among the given identified entities.

-Steps-
1. From the entities given by user, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, MUST be exactly one of the entity names from the provided entities list
- target_entity: name of the target entity, MUST be exactly one of the entity names from the provided entities list  
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

2. Return output in English as a single list of all the relationships identified in step 1. Use **{record_delimiter}** as the list delimiter.

3. When finished, output {completion_delimiter}

######################
-Important Constraints-
######################
- ONLY use entity names that appear EXACTLY in the provided entities list
- DO NOT create new entity names or modify existing ones
- DO NOT extract relationships involving entities not in the provided list
- Entity names are case-sensitive and must match exactly
- If no valid relationships exist between the provided entities, return an empty list

-Real Data-
######################
Entities: {entities}
Text: {input_text}
######################
Output:
"""

# For backward compatibility, maintain the original references
PROMPTS["hi_entity_extraction"] = PROMPTS["LEGACY_hi_entity_extraction"]
PROMPTS["hi_relation_extraction"] = PROMPTS["LEGACY_hi_relation_extraction"]


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entity_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "KG_EXTRACTION_CONTINUE"
] = """
MANY entities and relationships were missed in the last knowledge graph extraction.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score (1-10) indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

---Output---

Add them below using the same format:
""".strip()

PROMPTS[
    "entity_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS[
    "relation_if_loop_extraction"
] = """It appears some relations may have still been missed.  Answer YES | NO if there are still relations that need to be added.
"""

PROMPTS["KG_EXTRACTION"] = """
Given a text document, perform comprehensive knowledge graph extraction by identifying all entities and their relationships.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}], normal_entity means that doesn't belong to any other types.
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Identify all relationships among the identified entities. For each pair of related entities, extract:
- source_entity: name of the source entity (must be from identified entities)
- target_entity: name of the target entity (must be from identified entities)
- relationship_description: explanation of why the entities are related
- relationship_strength: numeric score (1-10) indicating relationship strength
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list with entities first, then relationships. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Example-
######################
Example:

Entity_types: [person, technology, organization, location, concept, event]
Text:
Dr. Sarah Chen stood before the quantum computing array at the MIT Research Laboratory, her hands trembling slightly as she prepared to initialize the revolutionary algorithm she had spent three years developing. The Quantum Neural Network project, funded by the Department of Defense, represented a breakthrough that could fundamentally alter cybersecurity protocols worldwide.

"Are you certain about this, Sarah?" asked Dr. Marcus Webb, the project director, his voice echoing through the sterile facility. "Once we activate the QNN, there's no going back. The implications for national security are enormous."

Sarah nodded firmly, her confidence unwavering despite the weight of responsibility. The algorithm had been tested countless times in simulation, but this would be its first real-world implementation. Behind them, a team of engineers from Quantum Dynamics Corporation monitored every parameter, knowing that success here could launch their company into a new era of technological dominance.

As the initialization sequence began, the laboratory fell silent except for the humming of cooling systems. This moment would later be known as the Genesis Protocol activation—a turning point that redefined the boundaries between human intelligence and artificial consciousness.

################
Output:
("entity"{tuple_delimiter}"Dr. Sarah Chen"{tuple_delimiter}"person"{tuple_delimiter}"Dr. Sarah Chen is a brilliant scientist who developed a revolutionary quantum algorithm over three years, serving as the lead researcher on the Quantum Neural Network project."){record_delimiter}
("entity"{tuple_delimiter}"Dr. Marcus Webb"{tuple_delimiter}"person"{tuple_delimiter}"Dr. Marcus Webb is the project director overseeing the Quantum Neural Network project, responsible for major decisions regarding its implementation."){record_delimiter}
("entity"{tuple_delimiter}"MIT Research Laboratory"{tuple_delimiter}"location"{tuple_delimiter}"MIT Research Laboratory is a high-tech research facility housing quantum computing arrays and serving as the testing ground for advanced technological projects."){record_delimiter}
("entity"{tuple_delimiter}"Quantum Neural Network"{tuple_delimiter}"technology"{tuple_delimiter}"The Quantum Neural Network is a revolutionary algorithm that represents a breakthrough in cybersecurity, combining quantum computing with neural network principles."){record_delimiter}
("entity"{tuple_delimiter}"Department of Defense"{tuple_delimiter}"organization"{tuple_delimiter}"The Department of Defense is the government agency providing funding for the Quantum Neural Network project due to its national security implications."){record_delimiter}
("entity"{tuple_delimiter}"Quantum Dynamics Corporation"{tuple_delimiter}"organization"{tuple_delimiter}"Quantum Dynamics Corporation is a technology company with engineers monitoring the QNN project, positioned to benefit from its success."){record_delimiter}
("entity"{tuple_delimiter}"Genesis Protocol"{tuple_delimiter}"event"{tuple_delimiter}"The Genesis Protocol activation represents the first real-world implementation of the Quantum Neural Network, marking a turning point in AI consciousness."){record_delimiter}
("entity"{tuple_delimiter}"Cybersecurity"{tuple_delimiter}"concept"{tuple_delimiter}"Cybersecurity refers to the protection of digital systems and networks, which could be fundamentally altered by the new quantum algorithm."){record_delimiter}
("relationship"{tuple_delimiter}"Dr. Sarah Chen"{tuple_delimiter}"Quantum Neural Network"{tuple_delimiter}"Dr. Sarah Chen is the creator and lead developer of the Quantum Neural Network algorithm, having spent three years perfecting it."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Dr. Marcus Webb"{tuple_delimiter}"Dr. Sarah Chen"{tuple_delimiter}"Dr. Marcus Webb serves as project director overseeing Dr. Sarah Chen's work, providing guidance and approval for major decisions."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"MIT Research Laboratory"{tuple_delimiter}"Quantum Neural Network"{tuple_delimiter}"MIT Research Laboratory provides the physical infrastructure and quantum computing arrays necessary for testing and implementing the QNN."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Department of Defense"{tuple_delimiter}"Quantum Neural Network"{tuple_delimiter}"The Department of Defense funds the Quantum Neural Network project due to its potential impact on national security and cybersecurity."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Quantum Dynamics Corporation"{tuple_delimiter}"Genesis Protocol"{tuple_delimiter}"Quantum Dynamics Corporation has engineers monitoring the Genesis Protocol activation, as its success could significantly benefit their business."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Quantum Neural Network"{tuple_delimiter}"Cybersecurity"{tuple_delimiter}"The Quantum Neural Network has the potential to fundamentally alter cybersecurity protocols and redefine digital protection methods."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Genesis Protocol"{tuple_delimiter}"Quantum Neural Network"{tuple_delimiter}"The Genesis Protocol represents the first real-world activation and implementation of the Quantum Neural Network system."{tuple_delimiter}10){completion_delimiter}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization",
    "person",
    "geo",
    "event",
    "concept",
    "technology",
]

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
