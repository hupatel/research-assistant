import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Research Paper Agent", layout="wide")
st.title("📄 Automated Research Paper Analyzer")

uploaded_file = st.file_uploader("Upload a research paper PDF", type=["pdf"])

user_goal = st.text_input(
    "Analysis goal",
    value="Summarize the paper and extract key findings"
)

if uploaded_file and user_goal:
    st.info("Uploading PDF...")

    upload_resp = requests.post(
        f"{API_BASE}/upload",
        files={"file": uploaded_file}
    )

    if upload_resp.status_code != 200:
        st.error("Upload failed")
        st.stop()

    paper_id = upload_resp.json()["paper_id"]
    st.success(f"Uploaded successfully (paper_id = {paper_id})")

    with st.spinner("Running LangGraph pipeline..."):
        analyze_resp = requests.post(
            f"{API_BASE}/analyze/{paper_id}",
            json={"user_goal": user_goal}
        )

    if analyze_resp.status_code != 200:
        st.error("Analysis failed")
        st.write(analyze_resp.text)
        st.stop()

    data = analyze_resp.json()
    report = data["final_report"]

    st.success("✅ Analysis completed!")

    st.header("📝 Summary")
    summary = report["outputs"].get("Extract Summary")
    if summary:
        st.subheader("Abstract Summary")
        st.write(summary.get("abstract_summary"))

        st.subheader("Overall Summary")
        st.write(summary.get("overall_summary"))

    findings = report["outputs"].get("Find Key Findings")
    if findings:
        st.header("🔬 Key Findings")
        for i, f in enumerate(findings.get("key_findings", []), 1):
            with st.expander(f"Finding {i}"):
                st.markdown(f"**Claim:** {f['claim']}")
                st.markdown(f"**Section:** {f.get('section')}")
                st.code(f["evidence"])

    review = report.get("review")
    if review:
        st.header("🧠 Reviewer")
        if review["grounded"]:
            st.success("All claims grounded in evidence.")
        else:
            st.warning("Some claims are not fully grounded.")

        for issue in review["issues"]:
            st.markdown(f"**{issue['severity'].upper()}**: {issue['issue']}")
            st.markdown(f"_Suggestion_: {issue['suggestion']}")

    with st.expander("⚙ Execution Logs"):
        for log in data.get("logs", []):
            st.write(log)

    st.download_button(
        "⬇ Download Full JSON Report",
        data=json.dumps(data, indent=2),
        file_name=f"{paper_id}.json",
        mime="application/json",
    )