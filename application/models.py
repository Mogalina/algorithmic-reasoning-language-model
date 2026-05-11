from sqlalchemy import Boolean, Column, Float, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    interviews = relationship("Interview", back_populates="user")


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    interviews = relationship("Interview", back_populates="company")
    interview_problems = relationship("InterviewProblem", back_populates="company")


class Interview(Base):
    __tablename__ = "interviews"

    id = Column(Integer, primary_key=True, index=True)
    days_until = Column(Integer, nullable=False)
    interview_date = Column(Date, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)

    user = relationship("User", back_populates="interviews")
    company = relationship("Company", back_populates="interviews")
    roadmaps = relationship("Roadmap", back_populates="interview")


class InterviewProblem(Base):
    __tablename__ = "interview_problems"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)
    description = Column(String, nullable=True)
    url = Column(String, nullable=False)
    topics = Column(String, nullable=True)
    acceptance_rate = Column(Float, nullable=True)
    frequency = Column(Float, nullable=True)
    preparation_days = Column(Integer, nullable=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)

    company = relationship("Company", back_populates="interview_problems")
    roadmaps = relationship("Roadmap", back_populates="problem")


class Roadmap(Base):
    __tablename__ = "roadmaps"

    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(Integer, ForeignKey("interviews.id"), nullable=False)
    problem_id = Column(Integer, ForeignKey("interview_problems.id"), nullable=False)
    is_completed = Column(Boolean, default=False)

    interview = relationship("Interview", back_populates="roadmaps")
    problem = relationship("InterviewProblem", back_populates="roadmaps")