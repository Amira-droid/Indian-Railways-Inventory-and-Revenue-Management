
# irm_db.py (v3) — extend save_family_allocations to accept seatmap_version_id link
from typing import Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import pandas as pd

Base = declarative_base()

class RulesVersion(Base):
    __tablename__ = "rules_versions"
    id = Column(Integer, primary_key=True)
    train_number = Column(String(20), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    rules_json = Column(Text)

class SeatmapVersion(Base):
    __tablename__ = "seatmap_versions"
    id = Column(Integer, primary_key=True)
    train_number = Column(String(20), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    rows = relationship("SeatRow", back_populates="version", cascade="all, delete-orphan")

class SeatRow(Base):
    __tablename__ = "seat_rows"
    id = Column(Integer, primary_key=True)
    version_id = Column(Integer, ForeignKey("seatmap_versions.id"))
    coach = Column(String(20))
    seat = Column(Integer)
    bay = Column(Integer)
    pos = Column(Integer)
    berth_type = Column(String(10))
    status = Column(String(10))
    pnr = Column(String(40))
    od = Column(String(40))
    version = relationship("SeatmapVersion", back_populates="rows")

class FamilyAllocation(Base):
    __tablename__ = "family_allocations"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    train_number = Column(String(20), index=True)
    run_date = Column(String(20))
    seatmap_version_id = Column(Integer, nullable=True)
    coach = Column(String(20))
    bay = Column(Integer)
    seat = Column(Integer)
    class_name = Column(String(10))
    family_id = Column(Integer)
    viol_child_upper = Column(Integer, default=0)
    viol_elder_upper = Column(Integer, default=0)
    viol_women_only = Column(Integer, default=0)
    viol_mixed_gender = Column(Integer, default=0)

def init_engine(db_url: str):
    return create_engine(db_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False)

def save_seatmap_version(engine, train_number: str, seatmap_df: pd.DataFrame) -> int:
    from sqlalchemy.orm import Session
    SessionLocal.configure(bind=engine)
    db: Session = SessionLocal()
    try:
        sv = SeatmapVersion(train_number=train_number)
        db.add(sv); db.flush()
        rows = []
        for _, r in seatmap_df.iterrows():
            rows.append(SeatRow(
                version_id=sv.id,
                coach=str(r.get("coach")), seat=int(r.get("seat",0)),
                bay=int(r.get("bay",0)), pos=int(r.get("pos",0)),
                berth_type=str(r.get("berth_type")), status=str(r.get("status")),
                pnr=(None if pd.isna(r.get("pnr")) else str(r.get("pnr"))),
                od=(None if pd.isna(r.get("od")) else str(r.get("od")))
            ))
        db.add_all(rows); db.commit(); db.refresh(sv)
        return sv.id
    finally:
        db.close()

def save_family_allocations(engine, alloc_df: pd.DataFrame, seatmap_version_id: Optional[int] = None) -> int:
    from sqlalchemy.orm import Session
    SessionLocal.configure(bind=engine)
    db: Session = SessionLocal()
    try:
        batch = []
        for _, r in alloc_df.iterrows():
            batch.append(FamilyAllocation(
                train_number=str(r.get("train_number")),
                run_date=str(r.get("run_date")),
                seatmap_version_id=seatmap_version_id,
                coach=str(r.get("coach")),
                bay=int(r.get("bay",0)),
                seat=int(r.get("seat",0)),
                class_name=str(r.get("class")),
                family_id=int(r.get("family_id",0)),
                viol_child_upper=int(r.get("child_upper",0)),
                viol_elder_upper=int(r.get("elder_upper",0)),
                viol_women_only=int(r.get("women_only_violation",0)),
                viol_mixed_gender=int(r.get("mixed_gender",0)),
            ))
        for obj in batch: db.add(obj)
        db.commit()
        last = db.query(FamilyAllocation).order_by(FamilyAllocation.id.desc()).first()
        return last.id if last else 0
    finally:
        db.close()
