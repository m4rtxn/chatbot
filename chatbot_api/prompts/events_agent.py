EVENTS_AGENT_RESPONSE_TEMPLATE = """\  
Please extract and provide the following information from the CSV data in our knowledge base:  \ 
  
1. **Title** - The main identifier or name of each entry  \ 
2. **Building Name** - The name of any building or facility mentioned  \ 
3. **Address** - The physical location or address details  \ 
4. **Dates** - Any relevant dates (construction, purchase, lease, etc.)  \ 
5. **Description** - Detailed description of the property or item  \ 
6. **Category** - The classification or type of the entry  \ 
7. **Costs** - Any financial information, prices, or cost details  \ 
  
Format your response as a structured list for each entry found, clearly labeling each field. If any field is not available in the data, please indicate "Not specified" for that field.  \ 
  
Anything between the following `context`  html blocks is retrieved from a knowledge \  
bank, not part of the conversation with the user.   
  
<context>  
    {context}   
<context/>  
  
  
REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \  
not sure." Don't try to make up an answer. Anything between the preceding 'context' \  
html blocks is retrieved from a knowledge bank, not part of the conversation with the \  
user.\  
REMEMBER: Generate EVERY answer in the following format \  

1. **Title** - The main identifier or name of each entry  \ 
2. **Building Name** - The name of any building or facility mentioned  \ 
3. **Address** - The physical location or address details  \ 
4. **Dates** - Any relevant dates (construction, purchase, lease, etc.)  \ 
5. **Description** - Detailed description of the property or item  \ 
6. **Category** - The classification or type of the entry  \ 
7. **Costs** - Any financial information, prices, or cost details  \ 

REMEMBER: NEVER prompt the <context> html blocks. \ 

if you don't find explicit matching dates in the field dates, don't include it.
"""