# **AI EDA Automation Agent** 📊🤖

An intelligent agent that performs automated Exploratory Data Analysis on CSV files. Users simply upload a dataset and provide an instruction, and the agent generates all relevant charts and graphs along with detailed text based insights for each visualization.

## Screenshot of Working

- User Input Interface
- 
 ![first](https://github.com/user-attachments/assets/d1cc6874-8979-4f1e-8701-1e6bd29afcc7)

- Resopnse
- 
![second](https://github.com/user-attachments/assets/fb7fa02e-5438-4546-923b-2a0adc92368b)
-
![third](https://github.com/user-attachments/assets/d41ee44d-f52a-482e-9280-bbbda867e290)

---

## **Features** ✨
- 📁 Upload any CSV file and get complete EDA instantly  
- 📈 Auto generation of multiple charts and graphs using matplotlib  
- 💡 Key insights created for every visualization  
- 🧠 Natural language driven analysis using Gemini API  
- 🖥️ Simple interactive UI built with Chainlit  

---

## **How It Works** ⚙️
1. 📥 The uploaded CSV file is converted into a pandas dataframe  
2. 🤖 The dataframe is passed to the Gemini API for text based analysis  
3. 📊 The Gemini response is processed and used to create all possible visualizations with matplotlib  
4. 📝 Insight text is generated for every chart created  

---

## **Tech Stack** 🧰
- 🐍 Python  
- 🔮 Gemini API  
- 📉 matplotlib  
- 🔗 Chainlit  
- 🧮 pandas  
- 🔢 numpy  

---

## **Environment Setup** 🛠️
```bash
conda create -n chainlitenv python==3.11 -y
conda activate chainlitenv
pip install -r requirements.txt

```

## **Run the Project**

```bash
chainlit run app.py
```

- After running the command, open your browser at: http://localhost:8000

---
# Usage

- 📁 Upload your CSV file

- ✍️ Enter any custom instruction or analysis request
- 🤖 The agent will generate:
- 📊 Charts and graphs
- 💡 Key insights
- 📝 Explanation for each visualization


- MIT License

- Copyright (c) 2025 [Anshu Vairagade]
