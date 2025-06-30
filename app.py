# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!
# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
    
# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!

    my_documents = [
        """Traditional Foods Across Cultures
Kajmak is a creamy dairy product traditional in the Balkans, especially Serbia and Montenegro. It is made by skimming cream from boiled milk and letting it ferment slightly, giving it a rich, salty taste. People eat kajmak with bread or grilled meats like ćevapi during family meals or celebrations. In Montenegro, kacamak is another traditional dish, made from cornmeal mash mixed with cheese or kajmak for a filling meal. In Japan, sushi began as preserved fish in fermented rice before evolving into fresh nigiri sushi in Edo times. In Ethiopia, injera isn’t just bread – it serves as a utensil to scoop stews, reflecting communal eating traditions. Mexico’s tamales, dating back to Aztec times, steam corn dough with fillings like pork or beans wrapped in husks. Croatia’s peka slow-cooks meat and vegetables under an iron bell with hot coals, creating smoky tenderness. Each dish shows local resources, preservation methods, and social values. Trying traditional foods when travelling teaches about identity, creativity, and what communities celebrate through their cuisine.""",

        """Languages and Worldviews
Language does more than label objects; it shapes how people think and what they notice. This idea is called linguistic relativity. For example, some Aboriginal Australian languages use cardinal directions like north, south, east, and west instead of left and right. Speakers must always know their orientation, and studies show they develop strong spatial awareness. Russian has two basic words for blue – goluboy for light blue and siniy for dark blue – affecting how speakers distinguish shades faster than English speakers. In Indonesia, the word gotong royong describes communal cooperation for the common good, a cultural concept without direct English translation. Learning other languages expands understanding of reality, culture, and social priorities. In Japan, the word amae describes a childlike dependence that shapes social interactions subtly. Such concepts show how language encodes values, social structures, and worldviews unique to a culture. When travelling or studying languages, noticing these differences helps appreciate diversity in thinking and behaviour across societies.""",

        """Festivals and Cultural Meaning
Festivals reflect cultural beliefs, values, and identities worldwide. In Japan, Hanami celebrates cherry blossoms as symbols of transient beauty and life’s impermanence rooted in Buddhist thought. People gather under blooming trees for picnics and music, enjoying fleeting beauty together. In Montenegro, the Mimosa Festival marks spring’s arrival in Herceg Novi with parades, seafood feasts, and music, celebrating the yellow mimosa flower that blooms along the Adriatic coast. Brazil’s Carnival blends Catholic and African traditions into street parties with samba dance, elaborate costumes, and social satire before Lent. India’s Holi festival uses coloured powders to welcome spring, break social boundaries, and strengthen community ties. Slovenia’s Kurentovanje involves fur-costumed figures dancing to chase away winter, rooted in pagan rituals. These festivals connect communities through ritual, food, and music, showing what cultures value and believe. When attending festivals while travelling, you witness how people express identity, preserve traditions, and renew social bonds through celebration.""",

        """Sustainable Travel Choices
Tourism fuels around 10% of global GDP and creates millions of jobs, but it can harm environments and communities if unmanaged. Sustainable travel minimises negative impacts while benefiting local economies. For example, staying in locally owned guesthouses keeps money in the community rather than large hotel chains. Using trains instead of flights cuts carbon emissions by up to 90% for short to medium distances. Travelling during off-peak seasons reduces crowding and strain on local infrastructure, improving experience for residents and visitors. Respecting wildlife viewing distances protects animals from stress and habitat disruption. Avoiding single-use plastics reduces waste in sensitive areas like beaches or forests. Eating local food supports farmers and preserves culinary heritage. Learning basic language phrases shows respect for local people and builds positive interactions. Sustainable choices require planning and awareness of environmental and cultural contexts. Responsible tourism ensures that future generations can enjoy destinations without losing their natural beauty, culture, or community wellbeing.""",

        """Etiquette and Everyday Culture
Learning etiquette before travelling prevents awkward moments and shows cultural sensitivity. In Montenegro and Serbia, greeting with a firm handshake and direct eye contact is polite, and addressing elders formally shows respect. Guests are often offered coffee or rakija as a sign of hospitality. Refusing it without explanation can seem rude. In Japan, people bow instead of shaking hands, and shoes are removed before entering homes. It is polite to say ‘itadakimasu’ before eating and ‘gochisousama’ after meals to thank for food. Middle Eastern cultures value offering guests tea or coffee, and it is polite to accept at least once. In India, eating with your right hand is the norm, while the left is considered unclean. Etiquette reflects deeper cultural values like hierarchy, collectivism, or humility. Observing local norms builds trust, avoids misunderstandings, and shows respect. When travellers adapt to local etiquette, they gain deeper cultural understanding and create more positive, respectful connections with local communities."""
    ]

    st.write("Documents loaded:", len(my_documents))

    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )

    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    st.write("Documents used for this answer:", docs)
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji ✈️ makes it more visually appealing
# This appears as the biggest text on your page
st.title("✈️ Travel & Culture Knowledge Hub")
st.markdown("### 🌍 Explore Travel & Culture")
st.markdown("*Your personal cultural knowledge assistant*")


# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
st.write("Welcome to my Travel & Culture Q&A app! Ask anything about foods, languages, festivals, or etiquette worldwide.")

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
question = st.text_input("What would you like to know about travel and culture?")

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
if st.button("Find My Answer", type="primary"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("Getting answer..."):
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**Answer:**")
        st.write(answer)
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("Please enter a question!")

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
with st.expander("About this Travel & Culture Q&A System"):
    st.write("""
    I created this app to answer questions about:
    - 🍲 Traditional foods and how they reflect culture
    - 🎉 Global festivals and their social meaning
    - 🗣️ Languages and worldviews
    - 🌱 Sustainable tourism practices
    - 🙏 Cultural etiquette around the world

    ✈️ Try asking about specific dishes, customs, festivals, or etiquette rules in different countries!
    """)

# TO RUN: Save as app.py, then type: streamlit run app.py
