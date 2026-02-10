
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract
import torch
import gradio as gr
from ultralytics import YOLO
from supabase import create_client, Client
import os

# --- Supabase Client Setup ---
SUPABASE_URL = "https://gwmjminqfyfxrzgaplzj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd3bWptaW5xZnlmeHJ6Z2FwbHpqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMjc2NjYsImV4cCI6MjA2NzgwMzY2Nn0.MzosL21U5BqmipTnSQaFnMmnwHCrmaaoGqMmxAOF258"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_interaction_to_supabase(query: str, response: str):
    try:
        data = {'query': query, 'response': response}
        res = supabase.table('interactions').insert(data).execute()
        if hasattr(res, "error") and res.error:
            print(f"Error saving to Supabase: {res.error.message}")
        else:
            print("Saved interaction to Supabase")
    except Exception as e:
        print(f"Exception during Supabase save: {e}")

# --- Model and Embedding Setup ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="D:/huggingface_cache")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- YOLO Apparatus Detection ---
yolo_model = YOLO("yolo.pt")  # Update with your model path

# --- Knowledge Base ---
documents = [
    # SCIENCE
    "Newton's first law states that an object will remain at rest or in uniform motion unless acted upon by an external force.",
    "Newton's second law states that Force equals mass times acceleration (F = ma).",
    "Newton's third law: For every action, there is an equal and opposite reaction.",
    "Acceleration is the rate of change of velocity per unit time.",
    "Mass is the quantity of matter in a body; weight is the force due to gravity on a mass.",
    "The law of conservation of momentum states that the total momentum before collision is equal to total momentum after collision.",
    "Work is said to be done when a force is applied and there is displacement in the direction of the force.",
    "Power is the rate of doing work and is measured in watts.",
    "The SI unit of energy is joule.",
    "Kinetic energy is the energy possessed by a body due to motion.",
    "Potential energy is the energy possessed due to position or configuration.",
    "The commercial unit of energy is kilowatt-hour (kWh).",
    "The human eye forms an inverted, real image on the retina.",
    "The least distance of distinct vision for a normal human eye is 25 cm.",
    "Myopia is a defect where distant objects cannot be seen clearly.",
    "Hypermetropia is a defect where nearby objects appear blurred.",
    "Photosynthesis occurs in chloroplasts using sunlight, water, and carbon dioxide.",
    "Mitochondria are known as the powerhouse of the cell.",
    "The heart has four chambers: two atria and two ventricles.",
    "Blood transports oxygen and nutrients and removes waste from the body.",
    "Transpiration is the loss of water through stomata in leaves.",
    "Reproduction can be sexual or asexual.",
    "Binary fission is a type of asexual reproduction seen in amoeba.",
    "Pollination is the transfer of pollen grains from the anther to stigma.",
    "Fertilization is the fusion of male and female gametes.",
    "Refraction is the bending of light when it passes from one medium to another.",
    "The angle of incidence is equal to the angle of reflection.",
    "The principal focus is the point where parallel rays meet after reflection or refraction.",
    "Concave mirrors are used in headlights.",
    "Convex mirrors are used as rear-view mirrors.",
    "The formula for magnification is height of image / height of object.",
    "The resistance of a conductor depends on its length, area, and material.",
    "Ohm's law: V = IR, where V is voltage, I is current, and R is resistance.",
    "The unit of current is ampere.",
    "Series circuits have only one path for current.",
    "Parallel circuits provide multiple paths.",
    "The fuse is a safety device used to prevent overloading.",
    "The Earth’s magnetic field resembles that of a bar magnet.",
    "A solenoid produces a uniform magnetic field when current flows through it.",
    "The right-hand thumb rule determines the direction of the magnetic field.",
    "Electromagnetic induction is the production of current by changing magnetic field.",
    "The ozone layer absorbs harmful UV rays from the sun.",
    "Global warming is the rise in average Earth temperatures due to greenhouse gases.",
    "Renewable energy sources include solar, wind, and hydro power.",
    "Non-renewable resources include coal, petroleum, and natural gas.",
    # MATH
    "The sum of the angles in a triangle is 180 degrees.",
    "The Pythagorean theorem: In a right triangle, a² + b² = c².",
    "The surface area of a sphere is 4πr².",
    "The volume of a cylinder is πr²h.",
    "The area of a circle is πr².",
    "The sum of the first n natural numbers is n(n+1)/2.",
    "The quadratic formula is x = [-b ± √(b² - 4ac)] / (2a).",
    "A pair of linear equations in two variables can be solved using substitution, elimination, or cross multiplication.",
    "Probability is the measure of likelihood of an event occurring.",
    "The mode is the number that occurs most frequently in a dataset.",
    "The mean is the sum of values divided by the number of values.",
    "The median is the middle value when data is arranged in order.",
    "An arithmetic progression (AP) has a common difference between terms.",
    "The nth term of an AP is a + (n-1)d.",
    "The sum of the first n terms of an AP is n/2[2a + (n-1)d].",
    "A polynomial is an expression with one or more terms.",
    "A linear polynomial has degree 1, a quadratic polynomial has degree 2.",
    "Euclid’s division lemma is used to prove the HCF of two numbers.",
    "Two lines are parallel if they never intersect.",
    "Two lines are perpendicular if they intersect at 90 degrees.",
    "The distance formula between two points is √[(x2 - x1)² + (y2 - y1)²].",
    "The coordinates of the midpoint are [(x1+x2)/2, (y1+y2)/2].",
    # HISTORY
    "The French Revolution started in 1789 and ended absolute monarchy in France.",
    "Bastille Day is celebrated on 14th July in France.",
    "The National Assembly abolished feudal privileges.",
    "Napoleon became emperor of France after the Revolution.",
    "The Industrial Revolution began in England in the 18th century.",
    "The Non-Cooperation Movement was launched by Gandhi in 1920.",
    "The Civil Disobedience Movement began with the Dandi March in 1930.",
    "The Quit India Movement was launched in 1942 demanding end of British rule.",
    "Mahatma Gandhi promoted non-violence and satyagraha.",
    "Subhash Chandra Bose led the Indian National Army against British rule.",
    "The Jallianwala Bagh massacre happened in 1919 in Amritsar.",
    "The Simon Commission was boycotted in India because it had no Indian members.",
    "The Rowlatt Act allowed detention without trial in India.",
    # CIVICS
    "Democracy is a form of government in which rulers are elected by the people.",
    "Universal adult franchise means every adult citizen can vote.",
    "The Indian Parliament consists of Lok Sabha and Rajya Sabha.",
    "The Prime Minister is the head of the government in India.",
    "The President is the ceremonial head of India.",
    "The Constitution of India came into effect on 26 January 1950.",
    "The judiciary in India is independent and impartial.",
    "Fundamental Rights are guaranteed by the Constitution to all citizens.",
    "Secularism means the state does not favor any religion.",
    "Federalism divides powers between central and state governments.",
    "The Election Commission of India conducts free and fair elections.",
    "The Right to Information Act empowers citizens to seek information from the government.",
    # GEOGRAPHY
    "India is a land of great physical diversity: mountains, plateaus, plains, deserts, and islands.",
    "The Himalayas form the northern mountain wall of India.",
    "The Northern Plains are formed by the Indus, Ganga, and Brahmaputra rivers.",
    "The Indian Peninsula is a part of the ancient landmass of Gondwanaland.",
    "India has six major physiographic divisions.",
    "India receives rainfall mainly from the southwest monsoon.",
    "The Thar Desert lies to the northwest of India.",
    "The Deccan Plateau is rich in minerals and black soil.",
    "The Western Ghats and Eastern Ghats are mountain ranges in southern India.",
    "India has a tropical monsoon climate.",
    "Natural vegetation varies from tropical evergreen to desert scrub.",
    "Forests are classified into reserved, protected, and unclassed forests.",
    # ECONOMICS
    "Development involves improving living standards and reducing poverty.",
    "Per capita income is the average income per person.",
    "The Human Development Index includes health, education, and income.",
    "Sustainable development means meeting needs without harming future generations.",
    "Unemployment is the condition where people are willing to work but cannot find jobs.",
    "Underemployment is when people work less than their capability.",
    "India has both formal and informal sectors of employment.",
    "Public sector enterprises are owned by the government.",
    "Private sector enterprises are owned by individuals or groups.",
    "Banks accept deposits and provide loans.",
    "Self Help Groups (SHGs) help small borrowers access credit.",
    "Globalization is the integration of economies around the world.",
    "Multinational corporations (MNCs) operate in multiple countries.",
    "Consumer rights include right to safety, information, and redressal.",
]

doc_embeddings = embedder.encode(documents, convert_to_tensor=True)

def add_to_knowledge_base(text):
    global documents, doc_embeddings
    documents.append(text)
    new_emb = embedder.encode([text], convert_to_tensor=True)
    doc_embeddings = torch.cat((doc_embeddings, new_emb), dim=0)

def semantic_search(query, top_k=5, threshold=0.25):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    sims = cosine_similarity(query_emb.unsqueeze(0).cpu(), doc_embeddings.cpu())[0]
    top_idxs = sorted(
        [(i, sims[i]) for i in range(len(sims)) if sims[i] >= threshold],
        key=lambda x: x[1],
        reverse=True
    )
    return [documents[i] for i, _ in top_idxs[:top_k]] if top_idxs else []

def answer_question_with_context(question, context_docs):
    if not context_docs:
        return "I'm sorry, I couldn't find relevant information to answer that."
    context = "\n".join(context_docs)
    prompt = (
        "You are a knowledgeable and concise CBSE Class 10 tutor. "
        "Read the provided context and answer the student's question directly and to the point. "
        "If the context is insufficient, state so politely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,           # Shorter, focused answers
        temperature=0.3,              # Less randomness, more to-the-point
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

def summarize_text_with_llm(text):
    prompt = (
        "Summarize the following note in a clear, concise way for a CBSE Class 10 student:\n\n"
        f"{text}\n\nSummary:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

def detect_apparatus(image, conf_threshold=0.25, iou_threshold=0.45):
    results = yolo_model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640
    )
    im_array = results[0].plot()
    im = Image.fromarray(im_array[..., ::-1])
    apparatus_names = [yolo_model.names[int(cls)] for cls in results[0].boxes.cls]
    return im, apparatus_names

def handle_query_or_image(query, image, detect_apparatus_mode=False, summarize_note=False):
    if image and detect_apparatus_mode:
        det_img, apparatus_list = detect_apparatus(image)
        apparatus_str = ", ".join(apparatus_list) if apparatus_list else "No apparatus detected."
        return f"Detected apparatus: {apparatus_str}", det_img
    elif image:
        text = extract_text_from_image(image)
        if summarize_note and text:
            summary = summarize_text_with_llm(text)
            return f"Summary of Note:\n{summary}", None
        elif text:
            add_to_knowledge_base(text)
        return "🖼️ Image processed and silently added to knowledge base.", None
    if not query:
        return "❗ Please enter a question or upload an image.", None
    add_to_knowledge_base(query)
    context = semantic_search(query)
    answer = answer_question_with_context(query, context)
    save_interaction_to_supabase(query, answer)
    return answer, None

iface = gr.Interface(
    fn=handle_query_or_image,
    inputs=[
        gr.Textbox(label="Ask a CBSE Class 10 Question"),
        gr.Image(type="pil", label="Upload Textbook or Lab Image"),
        gr.Checkbox(label="Detect Apparatus in Image"),
        gr.Checkbox(label="Summarize Note")  # Added for note summarization
    ],
    outputs=[
        gr.Textbox(label="TinyLlama's Answer / Apparatus Detection / Summary"),
        gr.Image(type="pil", label="Detection Result Image")
    ],
    title="📘 CBSE QA + Apparatus Detection + Note Summarizer (TinyLlama, Supabase Logging)",
    description="Ask questions, upload textbook/note images, detect apparatus, or summarize notes. All queries and answers are logged to Supabase.",
    allow_flagging="never",
)

iface.launch()

