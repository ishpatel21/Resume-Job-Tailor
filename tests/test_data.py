"""
Test data: Resume and job description with 90%+ match
"""

# High-match resume (Senior Python Backend Engineer)
RESUME_TEXT = """
JOHN SMITH
john.smith@email.com | (555) 123-4567 | linkedin.com/in/johnsmith

PROFESSIONAL SUMMARY
Senior Python Backend Engineer with 7+ years of professional backend development experience 
designing and implementing scalable microservices and cloud-based applications. 
Expert in developing REST and GraphQL APIs serving millions of requests daily.
Strong expertise in AWS cloud services, Docker containerization, and Kubernetes orchestration.
Hands-on experience with DevOps practices and serverless deployments on AWS Lambda.
Proven track record leading backend engineering teams, mentoring junior engineers, and architecting 
enterprise-grade systems. Strong background in machine learning implementations with TensorFlow, 
PyTorch, and Scikit-learn. Expert in production infrastructure management, CI/CD pipelines, and 
comprehensive testing strategies.

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, Java, SQL
Backend Frameworks: Django, FastAPI, Flask, Express, Spring Boot
Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
Cloud & DevOps: AWS (EC2, S3, Lambda, RDS), Azure, GCP, Kubernetes, Docker, Jenkins, GitLab CI/CD
Machine Learning: TensorFlow, PyTorch, Scikit-learn, Keras
APIs: REST, GraphQL, gRPC, Microservices Architecture
Testing: Pytest, Jest, JUnit, Unittest, RSpec
Version Control: Git, GitHub, GitLab
Leadership: Team Lead for 5-person backend team, Agile/Scrum management

PROFESSIONAL EXPERIENCE

Senior Backend Engineer | TechCorp Inc. | 2021 - Present
• Designed and implemented microservices architecture using Python FastAPI and Docker, managing infrastructure and mentor for junior engineers
• Deployed and managed Kubernetes clusters on AWS, reducing infrastructure costs by 40%
• Lead team of 5 engineers in developing REST and GraphQL APIs serving 10M+ requests/day
• Implemented CI/CD pipelines using Jenkins and GitLab for continuous deployment
• Architect and implemented machine learning recommendation system using TensorFlow and PyTorch
• Optimized PostgreSQL and MongoDB queries, improving API response time by 60%
• Mentor junior developers in clean code practices and design patterns (OOP, functional programming)
• Manage production systems and implement comprehensive testing framework with 90%+ code coverage

Backend Engineer | DataSystems Ltd. | 2018 - 2021
• Developed Django REST APIs for SaaS platform with 500K+ users
• Implemented Redis caching layer, improving response times from 2s to 200ms
• Built data warehouse using ETL pipelines for analytics and reporting
• Worked with data science team on machine learning models (scikit-learn, TensorFlow)
• Set up comprehensive testing framework with 90%+ code coverage using Pytest and Jest
• Implemented server-less functions on AWS Lambda for event processing

Full Stack Developer | StartupXYZ | 2016 - 2018
• Built Node.js Express APIs with TypeScript for web applications
• Developed React frontend components for SaaS application
• Implemented Git workflows and basic CI/CD with GitHub Actions
• Worked on problem-solving and optimization of critical system components

EDUCATION
BS in Computer Science | State University | 2016

CERTIFICATIONS
• AWS Certified Solutions Architect - Professional
• Docker Certified Associate
"""

# High-match job description
JOB_DESCRIPTION = """
Senior Python Backend Engineer

About Us
TechCorp is a fast-growing cloud infrastructure company serving enterprise clients globally.

Position Overview
We are seeking an experienced Senior Python Backend Engineer to lead our backend platform 
development and mentor our growing engineering team. You will architect scalable microservices, 
improve system reliability, and mentor junior developers.

Key Responsibilities
• Design and develop microservices using Python (FastAPI, Django) and REST/GraphQL APIs
• Architect scalable cloud solutions on AWS (EC2, S3, Lambda, RDS)
• Deploy and manage containerized applications using Docker and Kubernetes
• Implement and optimize database solutions (PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch)
• Build and maintain CI/CD pipelines using Jenkins and GitLab
• Lead code reviews, enforce clean code principles (OOP, design patterns, functional programming)
• Collaborate with data science team on machine learning implementations (TensorFlow, PyTorch, Scikit-learn)
• Mentor junior backend engineers and lead technical discussions
• Implement comprehensive testing strategies (Pytest, Jest, JUnit, Unittest, RSpec)
• Manage serverless deployments on AWS Lambda

Required Skills & Experience
• 6+ years of professional backend development experience
• Expert-level Python proficiency with FastAPI, Django, or Flask
• Strong understanding of microservices architecture and REST/GraphQL APIs
• Production experience with AWS cloud services (EC2, RDS, S3, Lambda)
• Hands-on experience with Docker, Kubernetes, and container orchestration
• Expertise in SQL and NoSQL databases (PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch)
• Strong Git workflows and experience with CI/CD (Jenkins, GitLab CI/CD)
• Solid testing foundation (Pytest, Jest, JUnit, Unittest)

Preferred Qualifications
• Experience with machine learning implementations (TensorFlow, PyTorch, Scikit-learn, Keras)
• DevOps and SRE experience
• Kubernetes cluster management at scale
• Data warehouse and ETL pipeline development
• Team leadership or mentoring experience
• Communication and problem-solving skills
• Agile/Scrum methodology experience
• Bachelor's degree in Computer Science or related field
• AWS or cloud certifications

Compensation & Benefits
• Competitive salary based on experience
• Remote or office location options
• Professional development budget
"""

if __name__ == "__main__":
    # Test with the matcher
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.matching.job_matcher import JobMatcher
    
    matcher = JobMatcher()
    results = matcher.match_resume_to_job(RESUME_TEXT, JOB_DESCRIPTION)
    
    print("=" * 70)
    print("TEST: HIGH-MATCH RESUME vs JOB DESCRIPTION")
    print("=" * 70)
    print(f"\n📊 Overall Match Score: {results['overall_score']}%")
    print(f"📍 Summary: {results['summary']}\n")
    
    print(f"Keyword Overlap: {results['keyword_overlap']['score']}%")
    print(f"  ✓ Matched: {len(results['keyword_overlap']['matched_keywords'])} keywords")
    print(f"  ✗ Missing: {len(results['keyword_overlap']['missing_keywords'])} keywords")
    
    print(f"\nTF-IDF Similarity: {results['tfidf_similarity']['score']}%")
    print(f"  Interpretation: {results['tfidf_similarity']['interpretation']}\n")
    
    if results['matching_keywords']:
        print(f"Top Matching Keywords: {', '.join(results['matching_keywords'][:15])}")
    
    if results['missing_keywords']:
        print(f"Missing Keywords (top 5): {', '.join(results['missing_keywords'][:5])}")
    
    print("\n" + "=" * 70)
