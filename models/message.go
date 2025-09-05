package models

import (
	"time"

	"github.com/google/uuid"
)

type MessageType string
type MessageStatus string
type UserFeedback string
type Metadata map[string]interface{}

const (
	MessageTypeConnected    MessageType = "connected"
	MessageTypeDisconnected MessageType = "disconnected"
	MessageTypeChat         MessageType = "chat"
	MessageTypeImage        MessageType = "image"
	MessageTypeVideo        MessageType = "video"
	MessageTypeRegenerate   MessageType = "regenerate"
	MessageTypeStop         MessageType = "stop"
	MessageTypeFeedback     MessageType = "feedback"
	MessageTypeResponse     MessageType = "response"
	MessageTypePing         MessageType = "ping"
)

const (
	MessageStatusApproved   MessageStatus = "approved"
	MessageStatusRejected   MessageStatus = "rejected"
	MessageStatusPending    MessageStatus = "pending"
	MessageStatusProcessing MessageStatus = "processing"
	MessageStatusSuccess    MessageStatus = "success"
	MessageStatusError      MessageStatus = "error"
)

const (
	FeedbackNone     UserFeedback = "none"
	FeedbackPositive UserFeedback = "positive"
	FeedbackNegative UserFeedback = "negative"
)

// Message model represents a user message and its associated response
type Message struct {
	ID                uuid.UUID     `json:"id" db:"id"`
	UserID            uuid.UUID     `json:"user_id" db:"user_id"`
	ConversationID    uuid.UUID     `json:"conversation_id" db:"conversation_id"`
	Language          string        `json:"language" db:"language" validate:"required,oneof=en zh kr jp ru"`
	AIMessage         string        `json:"ai_message" db:"ai_message" validate:"required,max=10000"`
	AIResponse        string        `json:"ai_response" db:"ai_response"`
	Message           string        `json:"message" db:"message" validate:"required,max=10000"`
	MessageEmbedding  *string       `json:"message_embedding,omitempty" db:"message_embedding"`
	Response          string        `json:"response" db:"response"`
	ResponseIndex     *int          `json:"response_index,omitempty" db:"response_index"`
	ResponseEmbedding *string       `json:"response_embedding,omitempty" db:"response_embedding"`
	Feedback          UserFeedback  `json:"feedback" db:"feedback"  validate:"required,oneof=none positive negative"`
	Metadata          Metadata      `json:"metadata,omitempty" db:"metadata"`
	Status            MessageStatus `json:"status" db:"status" validate:"required,oneof=approved rejected error"`
	CreatedAt         time.Time     `json:"created_at" db:"created_at"`
	UpdatedAt         time.Time     `json:"updated_at" db:"updated_at"`
	DeletedAt         *time.Time    `json:"deleted_at,omitempty" db:"deleted_at"`
}

type MessageMetadata struct {
	ProcessingTimeMs int64  `json:"processing_time_ms"`
	RejectedReason   string `json:"rejected_reason"`
}

// MessageStats model represents statistics for a conversation
type MessageStats struct {
	TotalMessages     int     `json:"total_messages"`
	AverageResponse   float64 `json:"average_response_time_ms"`
	AverageConfidence float64 `json:"average_confidence"`
}

type MessageSearchRequest struct {
	Query          string         `json:"query,omitempty"`
	ConversationID *uuid.UUID     `json:"conversation_id,omitempty"`
	StartDate      *time.Time     `json:"start_date,omitempty"`
	EndDate        *time.Time     `json:"end_date,omitempty"`
	Feedback       *UserFeedback  `json:"feedback,omitempty"`
	Status         *MessageStatus `json:"status,omitempty"`
	Limit          int            `json:"limit,omitempty" validate:"omitempty,min=1,max=100"`
	Offset         int            `json:"offset,omitempty" validate:"omitempty,min=0"`
}

type MessageSearchResponse struct {
	Messages []Message `json:"messages"`
	Total    int       `json:"total"`
	Offset   int       `json:"offset"`
	Limit    int       `json:"limit"`
}

type MessageRequest struct {
	Type           MessageType   `json:"type"`
	Language       *string       `json:"language"`
	UserID         uuid.UUID     `json:"user_id"`
	ConversationID *uuid.UUID    `json:"conversation_id,omitempty"`
	MessageID      *uuid.UUID    `json:"message_id,omitempty"`
	BotID          *string       `json:"bot_id,omitempty"`
	Message        string        `json:"message,omitempty"`
	Feedback       *UserFeedback `json:"feedback,omitempty"`
	Metadata       Metadata      `json:"metadata,omitempty"`
	Timestamp      time.Time     `json:"timestamp"`
}

type MessageResponse struct {
	Type           MessageType            `json:"type"`
	Status         MessageStatus          `json:"status"`
	Data           map[string]interface{} `json:"data,omitempty"`
	UserID         uuid.UUID              `json:"user_id"`
	ConversationID *uuid.UUID             `json:"conversation_id,omitempty"`
	MessageID      *uuid.UUID             `json:"message_id,omitempty"`
	BotID          *string                `json:"bot_id,omitempty"`
	Response       string                 `json:"string,omitempty"`
	ResponseIndex  *int                   `json:"response_index,omitempty"`
	Metadata       Metadata               `json:"metadata,omitempty"`
	Timestamp      time.Time              `json:"timestamp"`
}
